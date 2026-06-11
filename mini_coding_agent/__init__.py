"""Mini-Coding-Agent: a local multi-agent coding assistant using LangGraph and Ollama."""

from mini_coding_agent.agent import AgentState, MiniAgent
from mini_coding_agent.cli import build_welcome, main
from mini_coding_agent.context import SessionStore, WorkspaceContext
from mini_coding_agent.models import FakeModelClient, OllamaModelClient, OpenAiCompatibleClient
from mini_coding_agent.multi_agent import (
    MultiAgentRunner,
    SupervisorState,
    _parse_reviewer_verdict,
    build_supervisor_graph,
)

__all__ = [
    "AgentState",
    "FakeModelClient",
    "MiniAgent",
    "MultiAgentRunner",
    "OllamaModelClient",
    "OpenAiCompatibleClient",
    "SessionStore",
    "SupervisorState",
    "WorkspaceContext",
    "_parse_reviewer_verdict",
    "build_supervisor_graph",
    "build_welcome",
    "main",
]
