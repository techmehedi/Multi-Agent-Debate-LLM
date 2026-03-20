import gradio as gr
import os
import time
import random
from huggingface_hub import InferenceClient, repo_exists
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

DEFAULT_TEMP = 0.7

SYSTEM_MESSAGE = (
    "You are a helpful assistant participating in a multi-agent review board. "
    "Provide thoughtful, well-reasoned responses. When reviewing other agents' "
    "responses in later rounds, carefully consider their reasoning and update "
    "your answer if you find compelling arguments."
)


# ---------------------------------------------------------------------------
# Core debate logic (adapted from the research notebook for the HF Inference
# API -- replaces local transformers pipelines with InferenceClient calls)
# ---------------------------------------------------------------------------


def generate_answer(
    token: str,
    model: str,
    messages: list[dict],
    temperature: float,
) -> str:
    """Call the HF Inference API for a single agent turn, with retries."""
    client = InferenceClient(token=token, model=model)

    max_tries = 4
    base_sleep = 1.5
    last_exc: Exception | None = None

    for attempt in range(1, max_tries + 1):
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=temperature,
                top_p=0.9,
            )
            return response.choices[0].message.content

        except Exception as e:
            last_exc = e
            msg = repr(e).lower()

            # Treating these as transient: cold start, overloaded, gateway issues, timeouts
            transient = any(
                k in msg
                for k in [
                    "timeout",
                    "timed out",
                    "503",
                    "502",
                    "504",
                    "429",
                    "rate limited",
                    "too many requests",
                    "loading",
                    "overloaded",
                    "temporarily unavailable",
                    "service unavailable",
                    "gateway",
                ]
            )

            # If not transient, or we exhausted retries, re-raise
            if (not transient) or (attempt == max_tries):
                raise

            # Exponential backoff + small jitter
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.4)
            time.sleep(sleep_s)

    # Should never reach, but keeps type checkers happy
    raise last_exc if last_exc else RuntimeError("Unknown inference failure")
    
def get_hf_token() -> str | None:
    """
    Resolve a Hugging Face token from multiple possible sources.
    Priority:
    1. HF_TOKEN (Space secret)
    2. HUGGINGFACEHUB_API_TOKEN (older standard)
    3. HF_OAUTH_ACCESS_TOKEN (OAuth injection)
    """
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HF_OAUTH_ACCESS_TOKEN")
    )

def construct_review_message(other_responses: list[tuple[str, str]]) -> dict:
    """Build a peer-review prompt containing the other agents' latest answers."""
    if not other_responses:
        return {
            "role": "user",
            "content": "Please double-check your answer and provide your final response.",
        }

    parts = ["These are the responses to the problem from other agents:\n"]
    for label, resp in other_responses:
        parts.append(f"{label} response:\n```\n{resp}\n```\n")
    parts.append(
        "Using the reasoning from other agents as additional advice, update your answer. "
        "Examine your solution and that of the other agents step by step. Provide your final, updated response."
    )
    return {"role": "user", "content": "\n".join(parts)}


def handle_inference_error(error: Exception, model_name: str) -> str:
    """Return a user-friendly error string for common Inference API failures."""
    raw = repr(error)
    low = raw.lower()
    etype = type(error).__name__.lower()

    if "timeout" in etype or "timeout" in low:
        return (
            f"Request to '{model_name}' timed out. The model may be loading "
            "(cold start) or overloaded. Try again in a moment."
        )
    if "401" in raw or "403" in raw:
        return (
            f"Access denied for '{model_name}'. Visit the model page on "
            f"https://huggingface.co/{model_name} to accept its license/terms."
        )
    if "404" in raw:
        return f"Model '{model_name}' was not found on Hugging Face Hub."
    if "422" in raw:
        return (
            f"Model '{model_name}' does not support chat completion "
            "via the Inference API."
        )
    if "429" in raw:
        return "Rate limited. Please wait a moment and try again."
    if "402" in raw or "payment" in low or "credit" in low:
        return (
            "Out of Inference API credits. "
            "Check huggingface.co/settings/billing."
        )
    return f"Error with '{model_name}': {raw[:300]}"

def supports_chat_completion(model_id: str, token: str) -> tuple[bool, str]:
    try:
        client = InferenceClient(token=token, model=model_id)
        client.chat_completion(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
        )
        return True, ""
    except Exception as e:
        msg = handle_inference_error(e, model_id).strip().lower()

        if "does not support chat completion" in msg:
            return False, handle_inference_error(e, model_id)

        if "access denied" in msg:
            return False, handle_inference_error(e, model_id)

        # Everything else might be transient (cold start, 429, 5xx)
        return True, ""

def validate_model(model_id: str, token: str | None = None) -> tuple[bool, str]:
    """Return *(ok, error_message)* after checking the model exists on the Hub."""
    if not model_id or not model_id.strip():
        return False, "Model ID cannot be empty."
    model_id = model_id.strip()
    if model_id in DEFAULT_MODELS:
        return True, ""
    try:
        if not repo_exists(model_id, token=token):
            return False, f"Model '{model_id}' not found on Hugging Face Hub."
        return True, ""
    except Exception as exc:
        return False, f"Could not verify '{model_id}': {exc}"


def run_review_board(
    prompt: str,
    agent_configs: list[dict],
    num_rounds: int,
    token: str,
):
    """Generator yielding *(status_line, results_or_None)* tuples.

    *results* is ``None`` during processing and a dict mapping agent labels to
    their final-round response text on the very last yield.
    """
    num_agents = len(agent_configs)

    # Each agent gets its own conversation history
    agent_contexts: list[list[dict]] = [
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        for _ in range(num_agents)
    ]

    for round_num in range(num_rounds):
        tag = f"Round {round_num + 1}/{num_rounds}"
        yield f"**{tag}** -- Submitting requests...", None

        # After the first round, inject peer-review context
# After the first round, ONLY inject peer-review context into agent 0 (critic)
        if round_num > 0:
            for i in range(num_agents):
                others: list[tuple[str, str]] = []

                for j in range(num_agents):
                    if j == i:
                        continue

                    label = f"Agent {j + 1} (id={agent_configs[j]['id']})"
                    for msg in reversed(agent_contexts[j]):
                        if msg["role"] == "assistant":
                            others.append((label, msg["content"]))
                            break

                agent_contexts[i].append(construct_review_message(others))

        # Fan out requests concurrently
        futures: dict = {}
        max_workers = min(num_agents, 3)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i, cfg in enumerate(agent_configs):
                fut = pool.submit(
                    generate_answer,
                    token,
                    cfg["model"],
                    list(agent_contexts[i]),  # shallow copy for thread safety
                    cfg["temp"],
                )
                futures[fut] = i

            for fut in as_completed(futures):
                idx = futures[fut]
                model = agent_configs[idx]["model"]
                try:
                    text = fut.result()
                    agent_contexts[idx].append(
                        {"role": "assistant", "content": text}
                    )
                    yield (
                        f"**{tag}** -- Agent {idx + 1} (`{model}`) responded.",
                        None,
                    )
                except Exception as exc:
                    err = handle_inference_error(exc, model)
                    agent_contexts[idx].append(
                        {"role": "assistant", "content": f"[Error: {err}]"}
                    )
                    yield f"**{tag}** -- Agent {idx + 1} error: {err}", None

    # Collect each agent's final response
    results: dict[str, str] = {}
    for i, cfg in enumerate(agent_configs):
        last = "[No response generated]"
        for msg in reversed(agent_contexts[i]):
            if msg["role"] == "assistant":
                last = msg["content"]
                break
        results[f"Agent {i + 1} (id={cfg['id']}) -- {cfg['model']}"] = last

    yield (
        "**Complete!** Select an agent tab below to view their final response.",
        results,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.agent-header-wrap {
    padding: 0 !important;
    min-height: 0 !important;
    background: rgba(78, 70, 229, 1);
}
.agent-header {
    display: block;
    text-align: center;
    cursor: help;
}

.sidebar .group {
    margin-bottom: 8px !important;
}

*::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
*::-webkit-scrollbar-track {
    background: transparent;
}
*::-webkit-scrollbar-thumb {
    background: rgba(139, 92, 246, 0.45);
    border-radius: 4px;
}
*::-webkit-scrollbar-thumb:hover {
    background: rgba(139, 92, 246, 0.7);
}

/* Themed scrollbar -- Firefox */
* {
    scrollbar-width: thin;
    scrollbar-color: rgba(139, 92, 246, 0.45) transparent;
}
"""

with gr.Blocks(
    title="Multi-Agent Review Board",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
) as demo:

    # Shared state --------------------------------------------------------
    agents_state = gr.State([1, 2])
    next_id_state = gr.State(3)  # counter for the next ID to assign
    results_state = gr.State({})  # final responses dict (empty until a run)

    # ---- Sidebar --------------------------------------------------------
    with gr.Sidebar():
        gr.LoginButton()
        gr.Markdown("---")

        gr.Markdown("### Settings")
        num_rounds = gr.Slider(
            minimum=1,
            maximum=10,
            value=2,
            step=1,
            label="Rounds",
            info="Round 1 = independent answers.  Round 2+ = peer review.",
            interactive=True
        )

        gr.Markdown("---")
        gr.Markdown("### Agents")

        # Dynamic agent configuration rows
        @gr.render(inputs=agents_state)
        def render_agents(agent_ids):
            dropdowns: list = []
            sliders: list = []

            for idx, aid in enumerate(agent_ids):
                default_model = DEFAULT_MODELS[idx % len(DEFAULT_MODELS)]

                with gr.Group():
                    with gr.Row():
                        gr.HTML(
                            f'<span class="agent-header" title="Pick a model or type any HF model ID">'
                            f'<strong>Agent {idx + 1}</strong></span>',
                            elem_classes=["agent-header-wrap"],
                        )
                        if len(agent_ids) > 2:
                            del_btn = gr.Button(
                                "✕",
                                variant="stop",
                                size="sm",
                                min_width=36,
                                scale=0,
                                key=f"del-{aid}",
                            )

                            # Freeze `aid` via default-arg so each button deletes the correct agent
                            def _delete(current_ids, _target=aid):
                                return [x for x in current_ids if x != _target]

                            del_btn.click(_delete, agents_state, agents_state)

                    dd = gr.Dropdown(
                        choices=DEFAULT_MODELS,
                        value=default_model,
                        allow_custom_value=True,
                        label=None,
                        show_label=False,
                        key=f"model-{aid}",
                        interactive=True
                    )
                    temp = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=DEFAULT_TEMP,
                        step=0.1,
                        label="Temperature",
                        key=f"temp-{aid}",
                        interactive=True
                    )

                    dropdowns.append(dd)
                    sliders.append(temp)

            # ---- Wire the Run button (defined further below) ----
            def on_run(data):
                hf_token = get_hf_token()

                if not hf_token:
                    raise gr.Error(
                        "No Hugging Face token found.\n\n"
                        "Add an HF_TOKEN secret in the Space settings "
                        "or enable OAuth with model access."
                    )

                prompt = data[prompt_tb]
                rounds = data[num_rounds]

                if not prompt or not prompt.strip():
                    raise gr.Error("Please enter a prompt.")

                models = [data[dd] for dd in dropdowns]
                temps = [data[sl] for sl in sliders]
                
                agent_ids_local = list(agent_ids)  # stable IDs for this render
                
                configs: list[dict] = []
                for i, (aid, model, t) in enumerate(zip(agent_ids_local, models, temps)):
                    if not model or not model.strip():
                        raise gr.Error(f"Agent {i + 1}: please select or enter a model.")
                    model = model.strip()
                
                    if model not in DEFAULT_MODELS:
                        ok, err = validate_model(model, hf_token)
                        if not ok:
                            raise gr.Error(f"Agent {i + 1}: {err}")
                    
                        ok, err = supports_chat_completion(model, hf_token)
                        if not ok:
                            raise gr.Error(f"Agent {i + 1}: {err}")
                    else:
                        #  Verify defaults are chat-compatible
                        ok, err = supports_chat_completion(model, hf_token)
                        if not ok:
                            raise gr.Error(f"Agent {i + 1}: {err}")
                                    
                    configs.append({"id": aid, "model": model, "temp": float(t)})

                # Stream progress as an accumulating log
                log: list[str] = []
                for status_line, results in run_review_board(
                    prompt.strip(), configs, int(rounds), hf_token
                ):
                    log.append(status_line)
                    yield (
                        "\n\n".join(log),
                        results if results is not None else {},
                    )

            run_btn.click(
                on_run,
                inputs={prompt_tb, num_rounds} | set(dropdowns) | set(sliders),
                outputs=[status_md, results_state],
            )

        # "Add Agent" sits outside @gr.render so it stays at the bottom
        add_btn = gr.Button("+ Add Agent", variant="secondary", size="sm")

        def _add_agent(ids, nid):
            return ids + [nid], nid + 1

        add_btn.click(
            _add_agent,
            [agents_state, next_id_state],
            [agents_state, next_id_state],
        )

    # ---- Main area ------------------------------------------------------
    gr.Markdown("# Multi-Agent Review Board")
    gr.Markdown(
        "Configure your agents in the sidebar, enter a prompt, and let "
        "multiple AI models debate and refine their answers across rounds."
    )

    prompt_tb = gr.Textbox(
        label="Prompt",
        placeholder="Enter your question or prompt here...",
        lines=4,
    )
    run_btn = gr.Button("Run Review Board", variant="primary", size="lg")
    status_md = gr.Markdown("")

    # ---- Dynamic results tabs -------------------------------------------
    @gr.render(inputs=results_state)
    def render_results(results):
        if not results:
            return
        gr.Markdown("---")
        gr.Markdown("### Final Responses")
        with gr.Tabs():
            for name, response in results.items():
                with gr.TabItem(name):
                    gr.Markdown(response)


if __name__ == "__main__":
    demo.launch()
