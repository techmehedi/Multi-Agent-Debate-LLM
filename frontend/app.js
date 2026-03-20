// ================================================================
// Multi-Agent Review Board — Frontend Logic
// Calls HF Inference API directly from the browser. No backend needed.
// ================================================================

/* ── Constants ── */
const DEFAULT_MODELS = [
  "meta-llama/Llama-3.1-8B-Instruct",
  "Qwen/Qwen2.5-7B-Instruct",
  "google/gemma-3-27b-it",
  "google/gemma-3n-E4B-it",
  "Qwen/Qwen2.5-Coder-32B-Instruct",
  "meta-llama/Llama-3.2-3B-Instruct",
  "Qwen/QwQ-32B",
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
];

const SYSTEM_MESSAGE =
  "You are a helpful assistant participating in a multi-agent review board. " +
  "Provide thoughtful, well-reasoned responses. When reviewing other agents' " +
  "responses in later rounds, carefully consider their reasoning and update " +
  "your answer if you find compelling arguments.";

// Route through our Netlify Edge proxy — model is specified in the JSON body.
// The proxy always forwards to router.huggingface.co/hf-inference/v1/chat/completions
const HF_API_BASE = "/hf-api";

/* ── App State ── */
let state = {
  agents: [],
  nextId: 1,
  running: false,
  abortController: null,
  markdownEnabled: true,
};

/* ── DOM refs (assigned in init) ── */
let $tokenInput, $tokenToggle, $eyeIcon, $eyeOffIcon;
let $roundsSlider, $roundsValue;
let $agentsList, $agentCount;
let $addAgentBtn;
let $promptTextarea, $charCount;
let $runBtn, $runBtnText, $runIcon, $spinnerIcon, $stopBtn;
let $roundProgress, $roundLabel, $roundDots, $progressBar;
let $liveFeed, $liveIndicator, $logContainer;
let $resultsSection, $resultsGrid, $resultsTabs, $tabContent, $resultsControls, $toggleMarkdownBtn;
let $emptyState;

/* ══════════════════════════════════════════════════
   CORE DEBATE LOGIC
══════════════════════════════════════════════════ */

/**
 * Call the HF Inference API for a single agent turn.
 * Streams tokens via onChunk(fullTextSoFar, isDone).
 * Retries on transient errors with exponential backoff.
 */
async function generateAnswer(token, model, messages, temperature, onChunk, signal) {
  const endpoint = `${HF_API_BASE}/v1/chat/completions`;
  const maxTries = 4;
  const baseSleepMs = 1500;
  const transientKeywords = ["timeout", "timed out", "503", "502", "504", "429",
    "rate limit", "too many requests", "loading", "overloaded",
    "temporarily unavailable", "service unavailable", "gateway"];

  let lastErr = null;

  for (let attempt = 1; attempt <= maxTries; attempt++) {
    if (signal && signal.aborted) throw new DOMException("Aborted", "AbortError");

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          messages,
          max_tokens: 2048,
          temperature,
          top_p: 0.9,
          stream: true,
        }),
        signal,
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`HTTP ${res.status}: ${body}`);
      }

      // ─ Stream response ─
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") continue;
          try {
            const json = JSON.parse(data);
            const tok = json.choices?.[0]?.delta?.content;
            if (tok) {
              fullText += tok;
              if (onChunk) onChunk(fullText, false);
            }
          } catch (_) { /* skip malformed chunk */ }
        }
      }

      if (onChunk) onChunk(fullText, true);
      return fullText;

    } catch (err) {
      if (err.name === "AbortError") throw err;
      lastErr = err;
      const msg = (err.message || "").toLowerCase();
      const isTransient = transientKeywords.some(k => msg.includes(k));
      if (!isTransient || attempt === maxTries) throw err;
      const sleepMs = baseSleepMs * Math.pow(2, attempt - 1) + Math.random() * 400;
      await sleep(sleepMs);
    }
  }

  throw lastErr ?? new Error("Unknown inference failure");
}

/** Build the peer-review prompt message. */
function constructReviewMessage(otherResponses) {
  if (!otherResponses.length) {
    return {
      role: "user",
      content: "Please double-check your answer and provide your final response.",
    };
  }
  const parts = ["These are the responses to the problem from other agents:\n"];
  for (const [label, resp] of otherResponses) {
    parts.push(`${label} response:\n\`\`\`\n${resp}\n\`\`\`\n`);
  }
  parts.push(
    "Using the reasoning from other agents as additional advice, update your answer. " +
    "Examine your solution and that of the other agents step by step. Provide your final, updated response."
  );
  return { role: "user", content: parts.join("\n") };
}

/** Friendly error string from a fetch/API error. */
function friendlyError(err, model) {
  const raw = err.message || String(err);
  const low = raw.toLowerCase();
  if (low.includes("401") || low.includes("403"))
    return `Access denied for '${model}'. Check your HF token and accept the model license.`;
  if (low.includes("404"))
    return `Model '${model}' not found on Hugging Face Hub.`;
  if (low.includes("422"))
    return `Model '${model}' does not support chat completion via the Inference API.`;
  if (low.includes("429"))
    return "Rate limited — please wait a moment and try again.";
  if (low.includes("402") || low.includes("credit"))
    return "Out of Inference API credits — check huggingface.co/settings/billing.";
  if (low.includes("timeout") || low.includes("timed out"))
    return `Request to '${model}' timed out (cold start / overloaded). Try again in a moment.`;
  return `Error with '${model}': ${raw.slice(0, 200)}`;
}

/**
 * Orchestrate a full multi-round debate.
 * agentConfigs: [{id, model, temperature}, ...]
 */
async function runDebate(prompt, agentConfigs, numRounds, token) {
  const numAgents = agentConfigs.length;
  state.abortController = new AbortController();
  const signal = state.abortController.signal;

  // Each agent gets its own conversation history
  const contexts = agentConfigs.map(() => [
    { role: "system", content: SYSTEM_MESSAGE },
    { role: "user",   content: prompt },
  ]);

  showRoundProgress(numRounds);

  for (let round = 0; round < numRounds; round++) {
    if (signal.aborted) break;

    setRoundProgress(round, numRounds);
    appendLog(`── Round ${round + 1} of ${numRounds} ──`, "round");

    // After round 1: inject peer-review messages
    if (round > 0) {
      for (let i = 0; i < numAgents; i++) {
        const others = [];
        for (let j = 0; j < numAgents; j++) {
          if (j === i) continue;
          const last = [...contexts[j]].reverse().find(m => m.role === "assistant");
          if (last) others.push([`Agent ${j + 1} (${agentConfigs[j].model})`, last.content]);
        }
        contexts[i].push(constructReviewMessage(others));
      }
    }

    // Mark all agents as "thinking"
    agentConfigs.forEach((_, i) => setAgentStatus(i, "thinking"));

    // Fan out all agents concurrently
    const promises = agentConfigs.map((cfg, idx) =>
      (async () => {
        try {
          appendLog(`Agent ${idx + 1} (${shortModel(cfg.model)}) is responding…`);
          const text = await generateAnswer(
            token, cfg.model, [...contexts[idx]], cfg.temperature,
            (currentText, _isDone) => updateAgentStream(idx, currentText),
            signal
          );
          contexts[idx].push({ role: "assistant", content: text });
          updateAgentStream(idx, text, true);
          setAgentStatus(idx, "done");
          appendLog(`Agent ${idx + 1} responded ✓`, "success");
        } catch (err) {
          if (err.name === "AbortError") return;
          const msg = friendlyError(err, cfg.model);
          const errText = `[Error: ${msg}]`;
          contexts[idx].push({ role: "assistant", content: errText });
          updateAgentStream(idx, errText, true);
          setAgentStatus(idx, "error");
          appendLog(`Agent ${idx + 1} — ${msg}`, "error");
        }
      })()
    );

    await Promise.all(promises);
  }

  // Final: collect results
  const results = agentConfigs.map((cfg, i) => {
    const last = [...contexts[i]].reverse().find(m => m.role === "assistant");
    return { cfg, text: last ? last.content : "[No response]" };
  });

  setRoundProgress(numRounds, numRounds);
  appendLog("All rounds complete!", "success");
  setLiveIndicatorIdle();
  buildFinalTabs(results);
}

/* ══════════════════════════════════════════════════
   UI UPDATE HELPERS
══════════════════════════════════════════════════ */

function showRoundProgress(total) {
  $roundProgress.style.display = "";
  $roundDots.innerHTML = "";
  for (let i = 0; i < total; i++) {
    const d = document.createElement("span");
    d.className = "round-dot";
    d.dataset.idx = i;
    $roundDots.appendChild(d);
  }
  $progressBar.style.width = "0%";
}

function setRoundProgress(current, total) {
  $roundLabel.textContent = current < total
    ? `Round ${current + 1} of ${total}`
    : "Complete!";
  $progressBar.style.width = `${Math.round((current / total) * 100)}%`;
  $roundDots.querySelectorAll(".round-dot").forEach((d, i) => {
    d.className = "round-dot" + (i < current ? " done" : i === current ? " active" : "");
  });
}

function appendLog(msg, type = "") {
  const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const el = document.createElement("div");
  el.className = `log-entry${type ? " log-" + type : ""}`;
  el.innerHTML = `<span class="log-time">${now}</span><span class="log-msg">${escapeHtml(msg)}</span>`;
  $logContainer.appendChild(el);
  $logContainer.scrollTop = $logContainer.scrollHeight;
}

function setLiveIndicatorIdle() {
  $liveIndicator.classList.add("idle");
  $liveIndicator.innerHTML = `<span class="live-dot"></span> DONE`;
}

/** Update a specific agent card during streaming. */
function updateAgentStream(idx, text, isDone = false) {
  const card = document.getElementById(`agent-card-${idx}`);
  if (!card) return;
  const body = card.querySelector(".agent-result-body");

  if (state.markdownEnabled && isDone) {
    body.innerHTML = safeMarkdown(text);
    body.classList.add("markdown-rendered");
  } else {
    body.textContent = text;
    if (!isDone) {
      const cursor = document.createElement("span");
      cursor.className = "streaming-cursor";
      body.appendChild(cursor);
    }
  }
}

function setAgentStatus(idx, status) {
  const card = document.getElementById(`agent-card-${idx}`);
  if (!card) return;
  card.className = `agent-result-card ${status === "error" ? "has-error" : status}`;
  const badge = card.querySelector(".agent-status-badge");
  if (!badge) return;
  const labels = { thinking: "Thinking…", done: "Done ✓", error: "Error", waiting: "Waiting" };
  badge.className = `agent-status-badge status-${status}`;
  badge.textContent = labels[status] ?? status;
}

/** Build the final tabbed results view. */
function buildFinalTabs(results) {
  $resultsTabs.innerHTML = "";
  $tabContent.innerHTML = "";
  $resultsTabs.style.display = "flex";
  $tabContent.style.display = "";
  $resultsControls.style.display = "flex";

  results.forEach(({ cfg, text }, i) => {
    // Tab button
    const tab = document.createElement("button");
    tab.className = "result-tab" + (i === 0 ? " active" : "");
    tab.textContent = `Agent ${i + 1}`;
    tab.title = cfg.model;
    tab.onclick = () => {
      document.querySelectorAll(".result-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      showTabPanel(i, results);
    };
    $resultsTabs.appendChild(tab);
  });

  showTabPanel(0, results);
}

function showTabPanel(idx, results) {
  const { cfg, text } = results[idx];
  $tabContent.innerHTML = "";
  const panel = document.createElement("div");
  panel.className = "tab-content-panel";
  const isError = text.startsWith("[Error:");
  const modelShort = shortModel(cfg.model);

  panel.innerHTML = `
    <div class="agent-result-header" style="margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid var(--border);">
      <div class="agent-badge-large">${idx + 1}</div>
      <div class="agent-result-meta">
        <div class="agent-result-name">Agent ${idx + 1}</div>
        <div class="agent-result-model">${escapeHtml(cfg.model)}</div>
      </div>
      <span class="agent-status-badge ${isError ? 'status-error' : 'status-done'}">
        ${isError ? "Error" : `${modelShort} · temp ${cfg.temperature.toFixed(1)}`}
      </span>
    </div>
    <div class="agent-result-body markdown-rendered" id="tab-body-${idx}"></div>
  `;
  $tabContent.appendChild(panel);
  const body = document.getElementById(`tab-body-${idx}`);
  if (state.markdownEnabled && !isError) {
    body.innerHTML = safeMarkdown(text);
  } else {
    body.textContent = text;
    body.style.whiteSpace = "pre-wrap";
  }
}

/* ══════════════════════════════════════════════════
   AGENT MANAGEMENT
══════════════════════════════════════════════════ */

function addAgent(model, temperature = 0.7) {
  state.agents.push({ id: state.nextId++, model, temperature });
  renderAgentList();
}

function removeAgent(id) {
  if (state.agents.length <= 2) { showToast("Need at least 2 agents.", "error"); return; }
  state.agents = state.agents.filter(a => a.id !== id);
  renderAgentList();
}

function renderAgentList() {
  $agentsList.innerHTML = "";
  $agentCount.textContent = state.agents.length;

  state.agents.forEach((agent, idx) => {
    const card = document.createElement("div");
    card.className = "agent-card";
    card.id = `sidebar-agent-${agent.id}`;
    const canDelete = state.agents.length > 2;

    card.innerHTML = `
      <div class="agent-card-header">
        <div class="agent-number">${idx + 1}</div>
        <span class="agent-label">Agent ${idx + 1}</span>
        ${canDelete ? `<button class="btn-delete" title="Remove agent">✕</button>` : ""}
      </div>
      <select class="agent-model-select" data-id="${agent.id}">
        ${DEFAULT_MODELS.map(m => `<option value="${m}" ${m === agent.model ? "selected" : ""}>${shortModel(m)}</option>`).join("")}
      </select>
      <div class="agent-temp-row">
        <label>Temperature</label>
        <input type="range" min="0.1" max="2.0" step="0.1" value="${agent.temperature}" data-id="${agent.id}" class="temp-slider" />
        <span class="temp-val">${agent.temperature.toFixed(1)}</span>
      </div>
    `;

    // Model change
    card.querySelector(".agent-model-select").addEventListener("change", e => {
      const a = state.agents.find(x => x.id === +e.target.dataset.id);
      if (a) a.model = e.target.value;
    });

    // Temperature change
    const tempSlider = card.querySelector(".temp-slider");
    const tempVal   = card.querySelector(".temp-val");
    tempSlider.addEventListener("input", e => {
      const v = parseFloat(e.target.value);
      tempVal.textContent = v.toFixed(1);
      updateSliderFill(tempSlider, 0.1, 2.0);
      const a = state.agents.find(x => x.id === +e.target.dataset.id);
      if (a) a.temperature = v;
    });
    updateSliderFill(tempSlider, 0.1, 2.0);

    // Delete
    card.querySelector(".btn-delete")?.addEventListener("click", () => removeAgent(agent.id));

    $agentsList.appendChild(card);
  });
}

/* ══════════════════════════════════════════════════
   RUN / STOP
══════════════════════════════════════════════════ */

async function startRun() {
  const prompt = $promptTextarea.value.trim();
  // Strip invisible / non-ASCII chars that break HTTP headers (zero-width spaces,
  // smart quotes, BOM, etc. get silently copied from web pages and PDFs).
  const rawToken = $tokenInput.value;
  const token    = rawToken.replace(/[^\x20-\x7E]/g, "").trim();

  if (!token) { showToast("Please enter your Hugging Face token.", "error"); return; }

  // Warn the user if stripping changed their token (so they know to re-paste)
  if (token !== rawToken.trim()) {
    showToast("⚠️ Your token contained invisible characters — they've been stripped. If this keeps failing, re-copy your token from huggingface.co.", "error");
    $tokenInput.value = token; // Fix the input so subsequent runs work too
  }

  const rounds = parseInt($roundsSlider.value, 10);

  if (!prompt) { showToast("Please enter a prompt.", "error"); return; }

  state.running = true;
  setRunningUI(true);
  clearResults();

  // Build agent configs
  const agentConfigs = state.agents.map(a => ({ id: a.id, model: a.model, temperature: a.temperature }));

  // Show result cards immediately before responses arrive
  buildStreamingCards(agentConfigs);

  // Show panels
  $liveFeed.style.display = "";
  $resultsSection.style.display = "";
  $emptyState.style.display = "none";
  $logContainer.innerHTML = "";
  $liveIndicator.classList.remove("idle");
  $liveIndicator.innerHTML = `<span class="live-dot"></span> LIVE`;

  try {
    await runDebate(prompt, agentConfigs, rounds, token);
  } catch (err) {
    if (err.name !== "AbortError") {
      appendLog(`Fatal error: ${err.message}`, "error");
      showToast("Something went wrong — check the live feed.", "error");
    }
  } finally {
    state.running = false;
    setRunningUI(false);
  }
}

function stopRun() {
  state.abortController?.abort();
  appendLog("Stopped by user.", "error");
  setLiveIndicatorIdle();
  state.running = false;
  setRunningUI(false);
}

function setRunningUI(running) {
  $runBtn.disabled = running;
  $runIcon.style.display     = running ? "none" : "";
  $spinnerIcon.style.display = running ? ""     : "none";
  $runBtnText.textContent    = running ? "Running…" : "Run Review Board";
  $stopBtn.style.display     = running ? ""     : "none";
}

function buildStreamingCards(agentConfigs) {
  $resultsGrid.innerHTML = "";
  $resultsTabs.style.display = "none";
  $tabContent.style.display  = "none";
  $resultsControls.style.display = "none";

  agentConfigs.forEach((cfg, idx) => {
    const card = document.createElement("div");
    card.className = "agent-result-card";
    card.id = `agent-card-${idx}`;
    card.innerHTML = `
      <div class="agent-result-header">
        <div class="agent-badge-large">${idx + 1}</div>
        <div class="agent-result-meta">
          <div class="agent-result-name">Agent ${idx + 1}</div>
          <div class="agent-result-model">${escapeHtml(cfg.model)}</div>
        </div>
        <span class="agent-status-badge status-waiting">Waiting</span>
      </div>
      <div class="agent-result-body"><em style="color:var(--text-muted)">Waiting to respond…</em></div>
    `;
    $resultsGrid.appendChild(card);
  });
}

function clearResults() {
  $resultsGrid.innerHTML = "";
  $resultsTabs.innerHTML = "";
  $tabContent.innerHTML = "";
  $resultsTabs.style.display = "none";
  $tabContent.style.display = "none";
  $resultsControls.style.display = "none";
  $roundProgress.style.display = "none";
}

/* ══════════════════════════════════════════════════
   MARKDOWN TOGGLE
══════════════════════════════════════════════════ */

function toggleMarkdown() {
  state.markdownEnabled = !state.markdownEnabled;
  $toggleMarkdownBtn.classList.toggle("active", state.markdownEnabled);

  // Re-render all stream cards
  document.querySelectorAll(".agent-result-body").forEach(body => {
    const raw = body._rawText;
    if (!raw) return;
    if (state.markdownEnabled) {
      body.innerHTML = safeMarkdown(raw);
      body.classList.add("markdown-rendered");
    } else {
      body.textContent = raw;
      body.classList.remove("markdown-rendered");
    }
  });
}

/* ══════════════════════════════════════════════════
   UTILITY
══════════════════════════════════════════════════ */

function safeMarkdown(text) {
  if (typeof marked === "undefined") return escapeHtml(text);
  try { return marked.parse(text, { breaks: true, gfm: true }); }
  catch (_) { return escapeHtml(text); }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function shortModel(m) {
  const parts = m.split("/");
  return parts[parts.length - 1];
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function updateSliderFill(slider, min, max) {
  const pct = ((+slider.value - min) / (max - min)) * 100;
  slider.style.setProperty("--pct", pct + "%");
}

let toastTimer = null;
function showToast(msg, type = "info") {
  const $toast = document.getElementById("toast");
  $toast.textContent = msg;
  $toast.className = `toast ${type} show`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $toast.classList.remove("show"), 4000);
}

/* ══════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════ */

function init() {
  // Grab DOM refs
  $tokenInput        = document.getElementById("hf-token");
  $tokenToggle       = document.getElementById("token-toggle");
  $eyeIcon           = document.getElementById("eye-icon");
  $eyeOffIcon        = document.getElementById("eye-off-icon");
  $roundsSlider      = document.getElementById("rounds-slider");
  $roundsValue       = document.getElementById("rounds-value");
  $agentsList        = document.getElementById("agents-list");
  $agentCount        = document.getElementById("agent-count");
  $addAgentBtn       = document.getElementById("add-agent-btn");
  $promptTextarea    = document.getElementById("prompt-textarea");
  $charCount         = document.getElementById("char-count");
  $runBtn            = document.getElementById("run-btn");
  $runBtnText        = document.getElementById("run-btn-text");
  $runIcon           = document.getElementById("run-icon");
  $spinnerIcon       = document.getElementById("spinner-icon");
  $stopBtn           = document.getElementById("stop-btn");
  $roundProgress     = document.getElementById("round-progress");
  $roundLabel        = document.getElementById("round-label");
  $roundDots         = document.getElementById("round-dots");
  $progressBar       = document.getElementById("progress-bar");
  $liveFeed          = document.getElementById("live-feed");
  $liveIndicator     = document.getElementById("live-indicator");
  $logContainer      = document.getElementById("log-container");
  $resultsSection    = document.getElementById("results-section");
  $resultsGrid       = document.getElementById("results-grid");
  $resultsTabs       = document.getElementById("results-tabs");
  $tabContent        = document.getElementById("tab-content");
  $resultsControls   = document.getElementById("results-controls");
  $toggleMarkdownBtn = document.getElementById("toggle-markdown-btn");
  $emptyState        = document.getElementById("empty-state");

  // Initial agents
  addAgent(DEFAULT_MODELS[0], 0.7);
  addAgent(DEFAULT_MODELS[1], 0.7);

  // Rounds slider
  updateSliderFill($roundsSlider, 1, 10);
  $roundsSlider.addEventListener("input", () => {
    $roundsValue.textContent = $roundsSlider.value;
    updateSliderFill($roundsSlider, 1, 10);
  });

  // Token toggle
  $tokenToggle.addEventListener("click", () => {
    const isPw = $tokenInput.type === "password";
    $tokenInput.type = isPw ? "text" : "password";
    $eyeIcon.style.display    = isPw ? "none" : "";
    $eyeOffIcon.style.display = isPw ? ""     : "none";
  });

  // Token → enable run button when non-empty
  $tokenInput.addEventListener("input", () => {
    $runBtn.disabled = !$tokenInput.value.trim() || !$promptTextarea.value.trim();
  });

  // Prompt textarea char count & run button enable
  $promptTextarea.addEventListener("input", () => {
    const len = $promptTextarea.value.length;
    $charCount.textContent = `${len.toLocaleString()} char${len !== 1 ? "s" : ""}`;
    $runBtn.disabled = !$tokenInput.value.trim() || !$promptTextarea.value.trim();
  });

  // Add agent button
  $addAgentBtn.addEventListener("click", () => {
    const nextModel = DEFAULT_MODELS[state.agents.length % DEFAULT_MODELS.length];
    addAgent(nextModel, 0.7);
  });

  // Run / Stop
  $runBtn.addEventListener("click", startRun);
  $stopBtn.addEventListener("click", stopRun);

  // Prompt keyboard shortcut: Ctrl/Cmd + Enter to run
  $promptTextarea.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !$runBtn.disabled && !state.running) {
      e.preventDefault();
      startRun();
    }
  });

  // Markdown toggle
  $toggleMarkdownBtn.classList.toggle("active", state.markdownEnabled);
  $toggleMarkdownBtn.addEventListener("click", toggleMarkdown);

  // Initialize marked.js
  if (typeof marked !== "undefined") {
    marked.setOptions({ breaks: true, gfm: true });
  }

  // Intercept updateAgentStream to store raw text for markdown toggle
  const _orig = updateAgentStream;
  window.updateAgentStream = (idx, text, isDone) => {
    const card = document.getElementById(`agent-card-${idx}`);
    if (card) {
      const body = card.querySelector(".agent-result-body");
      if (body) body._rawText = text;
    }
    _orig(idx, text, isDone);
  };
}

document.addEventListener("DOMContentLoaded", init);
