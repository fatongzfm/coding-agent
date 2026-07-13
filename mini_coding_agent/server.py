"""FastAPI + WebSocket server for the observability dashboard."""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mini_coding_agent.config import load_config
from mini_coding_agent.context import SessionStore, WorkspaceContext
from mini_coding_agent.models import OllamaModelClient, OpenAiCompatibleClient
from mini_coding_agent.multi_agent import MultiAgentRunner
from mini_coding_agent.observability import WorkflowEvent, event_bus, metrics_collector

logger = logging.getLogger("mca.server")

STATIC_DIR = Path(__file__).parent / "web" / "static"

app = FastAPI(title="Mini-Coding-Agent Dashboard", version="0.1.0")

# Serve static assets if the directory exists.
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_executor = ThreadPoolExecutor(max_workers=2)


class RunRequest(BaseModel):
    user_message: str
    model: str | None = None
    host: str = "http://127.0.0.1:11434"
    api_key: str | None = None
    base_url: str | None = None
    approval: str = "auto"
    max_steps: int = 10
    max_new_tokens: int = 2048
    cwd: str = "."

    def model_post_init(self, __context):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MINI_CODING_AGENT_API_KEY")


def _build_runner(req: RunRequest):
    workspace = WorkspaceContext.build(req.cwd)
    store = SessionStore(Path(workspace.repo_root) / ".mini-coding-agent" / "sessions")
    cfg = load_config(None)
    model_cfg = cfg.get("model", {})
    multi_cfg = cfg.get("multi_agent", {})

    # Backfill request fields from config (config takes precedence over defaults)
    if req.api_key is None:
        req.api_key = model_cfg.get("api_key")
    if req.base_url is None:
        req.base_url = model_cfg.get("base_url", "https://api.openai.com/v1")
    if req.model is None:
        req.model = model_cfg.get("name", "gpt-4o-mini")

    temperature = model_cfg.get("temperature", 0.2)
    top_p = model_cfg.get("top_p", 0.9)
    timeout = model_cfg.get("timeout", 300)
    max_review_cycles = multi_cfg.get("max_review_cycles", 3)
    if req.api_key:
        model = OpenAiCompatibleClient(
            model=req.model,
            base_url=req.base_url or "https://api.openai.com/v1",
            api_key=req.api_key,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
    else:
        model = OllamaModelClient(
            model=req.model,
            host=req.host,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
    return MultiAgentRunner(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=req.approval,
        max_steps_planner=multi_cfg.get("max_steps_planner", req.max_steps),
        max_steps_coder=multi_cfg.get("max_steps_coder", req.max_steps),
        max_steps_tester=multi_cfg.get("max_steps_tester", req.max_steps),
        max_steps_reviewer=multi_cfg.get("max_steps_reviewer", req.max_steps),
        max_new_tokens=req.max_new_tokens,
        max_review_cycles=max_review_cycles,
    )


@app.post("/api/run")
async def api_run(req: RunRequest):
    """Trigger a multi-agent run from the dashboard."""
    global _main_loop
    if _main_loop is None:
        _main_loop = asyncio.get_running_loop()
    loop = asyncio.get_running_loop()

    def _run():
        try:
            runner = _build_runner(req)
            return runner.ask(req.user_message)
        except Exception as exc:
            logger.exception("run_failed")
            return {"_error": str(exc)}

    try:
        result = await loop.run_in_executor(_executor, _run)
        if isinstance(result, dict) and "_error" in result:
            return {"status": "error", "error": result["_error"]}
        return {"status": "ok", "result": result}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


class ConnectionManager:
    """Simple manager for active WebSocket connections."""

    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()

# Reference to the main asyncio event loop (set on first /api/run call).
_main_loop: asyncio.AbstractEventLoop | None = None


def _on_event(event: WorkflowEvent):
    """Callback invoked by the event bus; forwards to all WebSocket clients."""
    if _main_loop is not None:
        try:
            asyncio.run_coroutine_threadsafe(manager.broadcast(event.to_dict()), _main_loop)
        except Exception:
            pass
    else:
        # Fallback if we are still in the main thread before any request arrived.
        try:
            asyncio.create_task(manager.broadcast(event.to_dict()))
        except RuntimeError:
            pass


# Subscribe once at module load.
event_bus.subscribe(_on_event)


@app.get("/api/logs/{run_id}")
def api_logs(run_id: str):
    """Return persisted events for a given run."""
    events = event_bus.get_history(run_id)
    return {"run_id": run_id, "events": [ev.to_dict() for ev in events]}


@app.get("/api/metrics")
def api_metrics():
    """Return aggregated metrics for all runs."""
    return {"runs": metrics_collector.get_all_metrics()}


@app.get("/api/metrics/{run_id}")
def api_run_metrics(run_id: str):
    """Return metrics for a specific run."""
    return {"run_id": run_id, "metrics": metrics_collector.get_run_metrics(run_id)}


@app.get("/")
def index():
    """Serve the dashboard HTML."""
    html_path = STATIC_DIR / "index.html"
    if html_path.is_file():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard static files not found.</h1>")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            action = msg.get("action")
            if action == "ping":
                await websocket.send_json({"type": "pong"})
            elif action == "subscribe_run":
                run_id = msg.get("run_id")
                if run_id:
                    for ev in event_bus.get_history(run_id):
                        await websocket.send_json(ev.to_dict())
    except WebSocketDisconnect:
        manager.disconnect(websocket)
