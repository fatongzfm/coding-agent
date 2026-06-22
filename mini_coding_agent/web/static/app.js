/**
 * Mini-Coding-Agent Dashboard Frontend
 *
 * Connects via WebSocket to the FastAPI server and renders
 * the multi-agent workflow graph + event log in real time.
 */

const WS_URL = `ws://${window.location.host}/ws`;

// Node topology in display order.
const NODES = [
  { id: "START", label: "START", next: ["supervisor"] },
  { id: "supervisor", label: "Supervisor", next: ["planner"] },
  { id: "planner", label: "Planner", next: ["coder"] },
  { id: "coder", label: "Coder", next: ["tester"] },
  { id: "tester", label: "Tester", next: ["reviewer"] },
  { id: "reviewer", label: "Reviewer", next: ["END", "coder"] },
  { id: "END", label: "END", next: [] },
];

// State ------------------------------------------------------------------
let ws = null;
let currentRunId = null;
let reconnectTimer = null;

// DOM refs ---------------------------------------------------------------
const $status = document.getElementById("connection-status");
const $graph = document.getElementById("workflow-graph");
const $log = document.getElementById("event-log");
const $runInfo = document.getElementById("run-info");

// Build static graph ------------------------------------------------------
function buildGraph() {
  $graph.innerHTML = "";
  NODES.forEach((n, i) => {
    const wrapper = document.createElement("div");
    wrapper.className = "node-wrapper";
    wrapper.dataset.node = n.id;

    const node = document.createElement("div");
    node.className = "node idle";
    node.id = `node-${n.id}`;
    node.textContent = n.label;
    wrapper.appendChild(node);

    // Arrow to next (except last node)
    if (i < NODES.length - 1) {
      const arrow = document.createElement("div");
      arrow.className = "arrow";
      arrow.id = `arrow-${n.id}-${NODES[i + 1].id}`;
      wrapper.appendChild(arrow);
    }

    // Loop-back label for reviewer
    if (n.id === "reviewer") {
      const label = document.createElement("div");
      label.className = "loop-label";
      label.textContent = "needs_fix → Coder";
      wrapper.appendChild(label);
    }

    $graph.appendChild(wrapper);
  });
}

// Update node visual state ------------------------------------------------
function setNodeState(nodeId, state, badgeText = null) {
  const el = document.getElementById(`node-${nodeId}`);
  if (!el) return;
  el.className = `node ${state}`;

  // Remove old badge
  const old = el.querySelector(".badge");
  if (old) old.remove();

  if (badgeText) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = badgeText;
    el.appendChild(badge);
  }
}

function setArrowActive(from, to, active = true) {
  const el = document.getElementById(`arrow-${from}-${to}`);
  if (el) el.classList.toggle("active", active);
}

function resetGraph() {
  NODES.forEach((n) => setNodeState(n.id, "idle"));
  document.querySelectorAll(".arrow").forEach((a) => a.classList.remove("active"));
}

// Logging -----------------------------------------------------------------
function appendLog(event) {
  const entry = document.createElement("div");
  entry.className = "log-entry";

  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = event.timestamp ? event.timestamp.split("T")[1].slice(0, 12) : "--:--:--";

  const node = document.createElement("span");
  node.className = "node";
  node.textContent = event.node;

  const type = document.createElement("span");
  type.className = "type";
  type.textContent = event.event_type;

  const body = document.createElement("div");
  body.style.flex = "1";

  if (event.event_type === "llm_output" && event.payload?.raw != null) {
    // Pretty-print model raw output
    const meta = document.createElement("div");
    meta.style.color = "var(--muted)";
    meta.style.fontSize = "0.75rem";
    meta.textContent = `response_chars=${event.payload.response_chars || event.payload.raw.length}`;
    body.appendChild(meta);
    const pre = document.createElement("pre");
    pre.style.marginTop = "0.3rem";
    pre.style.background = "#0f1117";
    pre.style.padding = "0.4rem 0.6rem";
    pre.style.borderRadius = "4px";
    pre.style.maxHeight = "200px";
    pre.style.overflow = "auto";
    pre.textContent = event.payload.raw;
    body.appendChild(pre);
  } else if (event.event_type === "tool_result" && event.payload?.result != null) {
    const meta = document.createElement("div");
    meta.style.color = "var(--muted)";
    meta.style.fontSize = "0.75rem";
    meta.textContent = `tool=${event.payload.tool || "?"}  duration=${event.payload.duration_ms || "?"}ms`;
    body.appendChild(meta);
    const pre = document.createElement("pre");
    pre.style.marginTop = "0.3rem";
    pre.style.background = "#0f1117";
    pre.style.padding = "0.4rem 0.6rem";
    pre.style.borderRadius = "4px";
    pre.style.maxHeight = "120px";
    pre.style.overflow = "auto";
    pre.textContent = event.payload.result;
    body.appendChild(pre);
  } else {
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(event.payload, null, 2);
    body.appendChild(pre);
  }

  entry.appendChild(ts);
  entry.appendChild(node);
  entry.appendChild(type);
  entry.appendChild(body);

  $log.appendChild(entry);
  $log.scrollTop = $log.scrollHeight;
}

// Run info ----------------------------------------------------------------
function updateRunInfo(html) {
  $runInfo.innerHTML = html;
}

// WebSocket ---------------------------------------------------------------
function connect() {
  if (ws) return;
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    $status.textContent = "Connected";
    $status.className = "status connected";
    clearTimeout(reconnectTimer);
  };

  ws.onmessage = (msg) => {
    let ev;
    try {
      ev = JSON.parse(msg.data);
    } catch {
      return;
    }
    handleEvent(ev);
  };

  ws.onclose = () => {
    ws = null;
    $status.textContent = "Disconnected";
    $status.className = "status disconnected";
    reconnectTimer = setTimeout(connect, 2000);
  };

  ws.onerror = () => {
    ws?.close();
  };
}

// Event handling ----------------------------------------------------------
function handleEvent(ev) {
  appendLog(ev);

  switch (ev.event_type) {
    case "run_start": {
      currentRunId = ev.run_id;
      resetGraph();
      const req = ev.payload?.user_message || "(no message)";
      updateRunInfo(`
        <p><strong>Run ID:</strong> ${ev.run_id.slice(0, 8)}</p>
        <p><strong>Status:</strong> <span style="color:var(--accent)">Running</span></p>
        <p><strong>Request:</strong> ${escapeHtml(req)}</p>
      `);
      break;
    }

    case "node_start": {
      setNodeState(ev.node, "running");
      // Highlight incoming arrow from previous node
      const prev = findPreviousNode(ev.node);
      if (prev) setArrowActive(prev, ev.node, true);
      break;
    }

    case "node_end": {
      const isError = ev.payload?.error != null;
      setNodeState(ev.node, isError ? "error" : "success");
      if (ev.node === "reviewer") {
        const verdict = ev.payload?.verdict;
        if (verdict) setNodeState("reviewer", "success", verdict);
      }
      break;
    }

    case "run_end": {
      const status = ev.payload?.status || "finished";
      const color = status === "approved" ? "var(--success)" : "var(--warning)";
      updateRunInfo(`
        <p><strong>Run ID:</strong> ${ev.run_id.slice(0, 8)}</p>
        <p><strong>Status:</strong> <span style="color:${color}">${status.toUpperCase()}</span></p>
        <p><strong>Request:</strong> ${escapeHtml(ev.payload?.user_message || "")}</p>
        <p><strong>Answer:</strong> <pre style="margin-top:0.4rem;background:#0f1117;padding:0.5rem;border-radius:4px">${escapeHtml(ev.payload?.final_answer || "(none)")}</pre></p>
      `);
      setNodeState("END", status === "approved" ? "success" : "error");
      break;
    }
  }
}

function findPreviousNode(nodeId) {
  const idx = NODES.findIndex((n) => n.id === nodeId);
  if (idx > 0) return NODES[idx - 1].id;
  return null;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Task submission ---------------------------------------------------------
const $taskInput = document.getElementById("task-input");
const $taskSubmit = document.getElementById("task-submit");

async function submitTask() {
  const msg = $taskInput.value.trim();
  if (!msg) return;

  $taskSubmit.disabled = true;
  $taskSubmit.textContent = "运行中…";
  $log.innerHTML = ""; // clear previous log

  try {
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_message: msg }),
    });
    const data = await resp.json();
    if (data.status !== "ok") {
      const backendError = data.error || data.detail || JSON.stringify(data);
      appendLog({
        run_id: "system",
        timestamp: new Date().toISOString(),
        node: "system",
        event_type: "error",
        payload: { error: backendError },
      });
    }
  } catch (err) {
    appendLog({
      run_id: "system",
      timestamp: new Date().toISOString(),
      node: "system",
      event_type: "error",
      payload: { error: err.message },
    });
  } finally {
    $taskSubmit.disabled = false;
    $taskSubmit.textContent = "运行";
  }
}

$taskSubmit.addEventListener("click", submitTask);
$taskInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitTask();
});

// Init --------------------------------------------------------------------
buildGraph();
connect();

// Fallback: ensure log is scrollable even if CSS overflow-y is ignored
$log.addEventListener("wheel", (e) => {
  e.preventDefault();
  $log.scrollTop += e.deltaY;
}, { passive: false });

// Load persisted logs for current run after connection
async function loadPersistedLogs(run_id) {
  if (!run_id) return;
  try {
    const resp = await fetch(`/api/logs/${run_id}`);
    const data = await resp.json();
    if (data.events) {
      for (const ev of data.events) {
        handleEvent(ev);
      }
    }
  } catch (err) {
    console.error("failed to load persisted logs", err);
  }
}
