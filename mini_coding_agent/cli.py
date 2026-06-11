import argparse
import shutil
import sys
from pathlib import Path

from mini_coding_agent.agent import MiniAgent
from mini_coding_agent.context import WorkspaceContext, SessionStore, middle
from mini_coding_agent.models import OllamaModelClient, OpenAiCompatibleClient
from mini_coding_agent.multi_agent import MultiAgentRunner

HELP_TEXT = "/help, /memory, /session, /reset, /exit"
WELCOME_ART = (
    "/\\     /\\\\",
    "{  `---'  }",
    "{  O   O  }",
    "~~>  V  <~~",
    "\\  \\|/  /",
    "`-----'__",
)
HELP_DETAILS = "\n".join(
    [
        "Commands:",
        "/help    Show this help message.",
        "/memory  Show the agent's distilled working memory.",
        "/session Show the path to the saved session file.",
        "/reset   Clear the current session history and memory.",
        "/exit    Exit the agent.",
    ]
)


def build_welcome(agent, model, backend):
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):
        return "+" + char * (width - 2) + "+"

    def center(text):
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")
    rows = [center(text) for text in WELCOME_ART]
    rows.extend(
        [
            center("MINI CODING AGENT"),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("BACKEND", backend, "SESSION", agent.session["id"]),
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])


def build_agent(args):
    workspace = WorkspaceContext.build(args.cwd)
    store = SessionStore(Path(workspace.repo_root) / ".mini-coding-agent" / "sessions")
    if args.api_key:
        model = OpenAiCompatibleClient(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.ollama_timeout,
        )
    else:
        model = OllamaModelClient(
            model=args.model,
            host=args.host,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.ollama_timeout,
        )
    if args.mode == "multi":
        return MultiAgentRunner(
            model_client=model,
            workspace=workspace,
            session_store=store,
            approval_policy=args.approval,
            max_steps_planner=args.max_steps,
            max_steps_coder=args.max_steps,
        )
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return MiniAgent.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
        )
    return MiniAgent(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for Ollama models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--model", default="qwen3.5:4b", help="Model name for Ollama or API.")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama server URL.")
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible API.")
    parser.add_argument("--api-key", default=None, help="API key for online LLM. If set, uses OpenAI-compatible API instead of Ollama.")
    parser.add_argument("--config", default=None, help="Path to YAML config file. Defaults to config/default.yaml.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument(
        "--approval",
        choices=("ask", "auto", "never"),
        default="ask",
        help="Approval policy for risky tools; auto grants the model arbitrary command execution and file writes.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    parser.add_argument(
        "--mode",
        choices=("single", "multi"),
        default="single",
        help="Agent mode: single agent or multi-agent (planner + coder).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output.")
    parser.add_argument("--log-dir", default=".logs", help="Directory for log files.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    # Load config and apply fallback values for api_key / base_url
    from mini_coding_agent.config import load_config

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    if args.api_key is None:
        args.api_key = model_cfg.get("api_key")
    if args.base_url is None:
        args.base_url = model_cfg.get("base_url", "https://api.openai.com/v1")
    if args.model == "qwen3.5:4b" and "name" in model_cfg:
        args.model = model_cfg["name"]
    if args.temperature == 0.2 and "temperature" in model_cfg:
        args.temperature = model_cfg["temperature"]
    if args.top_p == 0.9 and "top_p" in model_cfg:
        args.top_p = model_cfg["top_p"]
    if args.max_new_tokens == 512 and "max_new_tokens" in model_cfg:
        args.max_new_tokens = model_cfg["max_new_tokens"]

    import logging
    from mini_coding_agent.logging_config import setup_logging

    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    setup_logging(level=log_level, log_dir=args.log_dir)

    agent = build_agent(args)

    backend = f"API:{args.base_url.replace('https://', '').replace('http://', '')}" if args.api_key else f"Ollama:{args.host}"
    print(build_welcome(agent, model=args.model, backend=backend))

    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            try:
                print(agent.ask(prompt))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    while True:
        try:
            user_input = input("\nmini-coding-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        print()
        try:
            print(agent.ask(user_input))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
