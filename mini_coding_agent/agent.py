import json
import logging
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from mini_coding_agent.context import (
    clip,
    middle,
    now,
    IGNORED_PATH_NAMES,
    MAX_HISTORY,
    MAX_TOOL_OUTPUT,
)

class AgentState(TypedDict):
    tool_steps: int
    attempts: int
    max_attempts: int
    user_message: str
    raw_output: str
    parse_kind: str
    parse_payload: dict | str | None
    tool_result: str | None
    final_answer: str | None


def _make_check_limits_node(agent):
    def check_limits_node(state: AgentState):
        return {}
    return check_limits_node


def _make_should_continue(agent):
    def should_continue(state: AgentState):
        if state.get("final_answer") is not None:
            return "end"
        if state["tool_steps"] >= agent.max_steps or state["attempts"] >= state["max_attempts"]:
            return "limit"
        return "continue"
    return should_continue


def _make_call_model_node(agent):
    logger = logging.getLogger("mca.agent")

    def call_model_node(state: AgentState):
        prompt = agent.prompt(state["user_message"])
        logger.debug("model_call role=%s prompt_chars=%d max_tokens=%d", agent.role, len(prompt), agent.max_new_tokens)
        raw = agent.model_client.complete(prompt, agent.max_new_tokens)
        logger.debug("model_response role=%s response_chars=%d", agent.role, len(raw))
        return {"raw_output": raw, "attempts": state["attempts"] + 1}
    return call_model_node


def _make_parse_output_node(agent):
    def parse_output_node(state: AgentState):
        kind, payload = MiniAgent.parse(state["raw_output"])
        return {"parse_kind": kind, "parse_payload": payload}
    return parse_output_node


def _make_handle_tool_node(agent):
    logger = logging.getLogger("mca.tools")

    def handle_tool_node(state: AgentState):
        payload = state["parse_payload"]
        name = payload.get("name", "")
        args = payload.get("args", {})
        t0 = time.perf_counter()
        result = agent.run_tool(name, args)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "tool_execution tool=%s duration_ms=%d result_chars=%d",
            name, duration_ms, len(result),
        )
        agent.record(
            {
                "role": "tool",
                "name": name,
                "args": args,
                "content": result,
                "created_at": now(),
            }
        )
        agent.note_tool(name, args, result)
        return {"tool_steps": state["tool_steps"] + 1, "tool_result": result}
    return handle_tool_node


def _make_handle_retry_node(agent):
    logger = logging.getLogger("mca.agent")

    def handle_retry_node(state: AgentState):
        payload = state["parse_payload"]
        agent.record({"role": "assistant", "content": payload, "created_at": now()})
        logger.debug("retry_notice role=%s", agent.role)
        return {}
    return handle_retry_node


def _make_handle_final_node(agent):
    logger = logging.getLogger("mca.agent")

    def handle_final_node(state: AgentState):
        payload = state["parse_payload"]
        final = (payload or state["raw_output"]).strip()
        agent.record({"role": "assistant", "content": final, "created_at": now()})
        agent.remember(agent.session["memory"]["notes"], clip(final, 220), 5)
        logger.info("final_answer role=%s answer_chars=%d", agent.role, len(final))
        return {"final_answer": final}
    return handle_final_node


def _make_handle_limit_node(agent):
    logger = logging.getLogger("mca.agent")

    def handle_limit_node(state: AgentState):
        if state["attempts"] >= state["max_attempts"] and state["tool_steps"] < agent.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
        else:
            final = "Stopped after reaching the step limit without a final answer."
        agent.record({"role": "assistant", "content": final, "created_at": now()})
        logger.info("limit_reached role=%s reason=%s", agent.role, "malformed" if "malformed" in final else "steps")
        return {"final_answer": final}
    return handle_limit_node


def _build_agent_graph(agent):
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(AgentState)

    builder.add_node("check_limits", _make_check_limits_node(agent))
    builder.add_node("call_model", _make_call_model_node(agent))
    builder.add_node("parse_output", _make_parse_output_node(agent))
    builder.add_node("handle_tool", _make_handle_tool_node(agent))
    builder.add_node("handle_retry", _make_handle_retry_node(agent))
    builder.add_node("handle_final", _make_handle_final_node(agent))
    builder.add_node("handle_limit", _make_handle_limit_node(agent))

    builder.add_edge(START, "check_limits")
    builder.add_conditional_edges(
        "check_limits",
        _make_should_continue(agent),
        {"continue": "call_model", "limit": "handle_limit", "end": END},
    )
    builder.add_edge("call_model", "parse_output")
    builder.add_conditional_edges(
        "parse_output",
        lambda state: state["parse_kind"],
        {"tool": "handle_tool", "retry": "handle_retry", "final": "handle_final"},
    )
    builder.add_edge("handle_tool", "check_limits")
    builder.add_edge("handle_retry", "check_limits")
    builder.add_edge("handle_final", END)
    builder.add_edge("handle_limit", END)

    return builder.compile()


class MiniAgent:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        role="coder",
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.role = role
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": {"task": "", "files": [], "notes": []},
        }
        self.tools = self.build_tools()
        self.prefix = self.build_prefix()
        self.session_path = self.session_store.save(self.session)
        self.graph = _build_agent_graph(self)

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    ###############################################
    #### 3) Structured Tools And Permissions ######
    ###############################################
    def build_tools(self):
        tools = {
            "list_files": {
                "schema": {"path": "str='.'"},
                "risky": False,
                "description": "List files in the workspace.",
                "run": self.tool_list_files,
            },
            "read_file": {
                "schema": {"path": "str", "start": "int=1", "end": "int=200"},
                "risky": False,
                "description": "Read a UTF-8 file by line range.",
                "run": self.tool_read_file,
            },
            "search": {
                "schema": {"pattern": "str", "path": "str='.'"},
                "risky": False,
                "description": "Search the workspace with rg or a simple fallback.",
                "run": self.tool_search,
            },
            "run_shell": {
                "schema": {"command": "str", "timeout": "int=20"},
                "risky": True,
                "description": "Run a shell command in the repo root.",
                "run": self.tool_run_shell,
            },
        }
        if self.role == "coder":
            tools["write_file"] = {
                "schema": {"path": "str", "content": "str"},
                "risky": True,
                "description": "Write a text file.",
                "run": self.tool_write_file,
            }
            tools["patch_file"] = {
                "schema": {"path": "str", "old_text": "str", "new_text": "str"},
                "risky": True,
                "description": "Replace one exact text block in a file.",
                "run": self.tool_patch_file,
            }
        if self.depth < self.max_depth:
            tools["delegate"] = {
                "schema": {"task": "str", "max_steps": "int=3"},
                "risky": False,
                "description": "Ask a bounded read-only child agent to investigate.",
                "run": self.tool_delegate,
            }
        return tools

    ############################################
    #### 2) Prompt Shape And Cache Reuse #######
    ############################################
    def build_prefix(self):
        tool_lines = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        tool_text = "\n".join(tool_lines)
        examples = "\n".join(
            [
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
                '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
                '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                "<final>Done.</final>",
            ]
        )
        rules = "\n".join([
            "- Use tools instead of guessing about the workspace.",
            "- Return exactly one <tool>...</tool> or one <final>...</final>.",
            "- Tool calls must look like:",
            '  <tool>{"name":"tool_name","args":{...}}</tool>',
            "- For write_file and patch_file with multi-line text, prefer XML style:",
            '  <tool name="write_file" path="file.py"><content>...</content></tool>',
            "- Final answers must look like:",
            "  <final>your answer</final>",
            "- Never invent tool results.",
            "- Keep answers concise and concrete.",
            "- If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.",
            "- Before writing tests for existing code, read the implementation first.",
            "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
            "- New files should be complete and runnable, including obvious imports.",
            "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.",
            "- Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={}.",
        ])
        if self.role == "planner":
            identity = (
                "You are a Software Architect Planner, a specialized agent that analyzes "
                "requirements and produces detailed implementation plans. Do NOT write code. "
                "Only produce a plan. Use read_file, search, and list_files to understand "
                "the existing codebase before planning."
            )
        elif self.role == "tester":
            identity = (
                "You are a Quality Assurance Tester, a specialized read-only agent that "
                "validates code correctness, runs tests, checks edge cases, and reports findings. "
                "Do NOT modify any files. Use run_shell, read_file, and search to inspect and test "
                "the implementation. Return a concise test report with pass/fail status and a list "
                "of issues found."
            )
        elif self.role == "reviewer":
            identity = (
                "You are a Senior Code Reviewer, a specialized read-only agent that performs "
                "code review, security checks, and best-practice validation. Do NOT modify any files. "
                "Use read_file and search to inspect the implementation and the test report. "
                "Return your verdict as either:\n"
                "  <final>approved</final>\n"
                "or\n"
                "  <final>needs_fix: [detailed feedback]</final>"
            )
        else:
            identity = (
                "You are Mini-Coding-Agent, a small local coding agent running through Ollama. "
                "Implement the given plan by writing complete, runnable code with tests."
            )
        return "\n\n".join([
            identity,
            "Rules:\n" + rules,
            "Tools:\n" + tool_text,
            "Valid response examples:\n" + examples,
            self.workspace.text(),
        ])

    def memory_text(self):
        memory = self.session["memory"]
        notes = "\n".join(f"- {note}" for note in memory["notes"]) or "- none"
        return "\n".join([
            "Memory:",
            f"- task: {memory['task'] or '-'}",
            f"- files: {', '.join(memory['files']) or '-'}",
            "- notes:",
            notes,
        ])

    #####################################################
    #### 4) Context Reduction And Output Management #####
    #####################################################
    def history_text(self):
        history = self.session["history"]
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] in ("write_file", "patch_file"):
                path = str(item["args"].get("path", ""))
                seen_reads.discard(path)
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if item["role"] == "tool":
                limit = 900 if recent else 180
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    ########################################################
    #### 2) Prompt Shape And Cache Reuse (Continued) #######
    ########################################################
    def prompt(self, user_message):
        return "\n\n".join([
            self.prefix,
            self.memory_text(),
            "Transcript:\n" + self.history_text(),
            "Current user request:\n" + user_message,
        ])

    ###############################################
    #### 5) Session Memory (Continued) ###########
    ###############################################
    def record(self, item):
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def note_tool(self, name, args, result):
        memory = self.session["memory"]
        path = args.get("path")
        if name in {"read_file", "write_file", "patch_file"} and path:
            self.remember(memory["files"], str(path), 8)
        note = f"{name}: {clip(str(result).replace(chr(10), ' '), 220)}"
        self.remember(memory["notes"], note, 5)

    def ask(self, user_message):
        memory = self.session["memory"]
        if not memory["task"]:
            memory["task"] = clip(user_message.strip(), 300)
        self.record({"role": "user", "content": user_message, "created_at": now()})

        max_attempts = max(self.max_steps * 3, self.max_steps + 4)
        initial_state = {
            "tool_steps": 0,
            "attempts": 0,
            "max_attempts": max_attempts,
            "user_message": user_message,
            "raw_output": "",
            "parse_kind": "",
            "parse_payload": None,
            "tool_result": None,
            "final_answer": None,
        }
        final_state = self.graph.invoke(initial_state) # type: ignore
        return final_state["final_answer"]

    #############################################################
    #### 3) Structured Tools, Validation, And Permissions #######
    #############################################################
    def run_tool(self, name, args):
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            self.validate_tool(name, args)
        except Exception as exc:
            example = self.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            return message
        if self.repeated_tool_call(name, args):
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        if tool["risky"] and not self.approve(name, args):
            return f"error: approval denied for {name}"
        try:
            return clip(tool["run"](args))
        except Exception as exc:
            return f"error: tool {name} failed: {exc}"

    def repeated_tool_call(self, name, args):
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    def tool_example(self, name):
        examples = {
            "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
            "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
            "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
            "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
            "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
        }
        return examples.get(name, "")

    def validate_tool(self, name, args):
        args = args or {}

        if name == "list_files":
            path = self.path(args.get("path", "."))
            if not path.is_dir():
                raise ValueError("path is not a directory")
            return

        if name == "read_file":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            start = int(args.get("start", 1))
            end = int(args.get("end", 200))
            if start < 1 or end < start:
                raise ValueError("invalid line range")
            return

        if name == "search":
            pattern = str(args.get("pattern", "")).strip()
            if not pattern:
                raise ValueError("pattern must not be empty")
            self.path(args.get("path", "."))
            return

        if name == "run_shell":
            command = str(args.get("command", "")).strip()
            if not command:
                raise ValueError("command must not be empty")
            timeout = int(args.get("timeout", 20))
            if timeout < 1 or timeout > 120:
                raise ValueError("timeout must be in [1, 120]")
            return

        if name == "write_file":
            path = self.path(args["path"])
            if path.exists() and path.is_dir():
                raise ValueError("path is a directory")
            if "content" not in args:
                raise ValueError("missing content")
            return

        if name == "patch_file":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            old_text = str(args.get("old_text", ""))
            if not old_text:
                raise ValueError("old_text must not be empty")
            if "new_text" not in args:
                raise ValueError("missing new_text")
            text = path.read_text(encoding="utf-8")
            count = text.count(old_text)
            if count != 1:
                raise ValueError(f"old_text must occur exactly once, found {count}")
            return

        if name == "delegate":
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")
            task = str(args.get("task", "")).strip()
            if not task:
                raise ValueError("task must not be empty")
            return

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        raw = str(raw)
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = MiniAgent.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", MiniAgent.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", MiniAgent.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", MiniAgent.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", MiniAgent.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = MiniAgent.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", MiniAgent.retry_notice()
        if "<final>" in raw:
            final = MiniAgent.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", MiniAgent.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", MiniAgent.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = MiniAgent.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = MiniAgent.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        self.session["history"] = []
        self.session["memory"] = {"task": "", "files": [], "notes": []}
        self.session_store.save(self.session)

    def path_is_within_root(self, resolved):
        probe = resolved
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        for candidate in (probe, *probe.parents):
            try:
                if candidate.samefile(self.root):
                    return True
            except OSError:
                continue
        return False

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if not self.path_is_within_root(resolved):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved

    def tool_list_files(self, args):
        path = self.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        entries = [
            item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
            if item.name not in IGNORED_PATH_NAMES
        ]
        lines = []
        for entry in entries[:200]:
            kind = "[D]" if entry.is_dir() else "[F]"
            lines.append(f"{kind} {entry.relative_to(self.root)}")
        return "\n".join(lines) or "(empty)"

    def tool_read_file(self, args):
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
        return f"# {path.relative_to(self.root)}\n{body}"

    def tool_search(self, args):
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        path = self.path(args.get("path", "."))

        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or result.stderr.strip() or "(no matches)"

        matches = []
        files = [path] if path.is_file() else [
            item for item in path.rglob("*")
            if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(self.root).parts)
        ]
        for file_path in files:
            for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if pattern.lower() in line.lower():
                    matches.append(f"{file_path.relative_to(self.root)}:{number}:{line}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def tool_run_shell(self, args):
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        result = subprocess.run(
            command,
            cwd=self.root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return "\n".join(
            [
                f"exit_code: {result.returncode}",
                "stdout:",
                result.stdout.strip() or "(empty)",
                "stderr:",
                result.stderr.strip() or "(empty)",
            ]
        )

    def tool_write_file(self, args):
        path = self.path(args["path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(self.root)} ({len(content)} chars)"

    def tool_patch_file(self, args):
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
        return f"patched {path.relative_to(self.root)}"

    ###################################################
    #### 6) Delegation And Bounded Subagents ##########
    ###################################################
    def tool_delegate(self, args):
        if self.depth >= self.max_depth:
            raise ValueError("delegate depth exceeded")
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        child = MiniAgent(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            approval_policy="never",
            max_steps=int(args.get("max_steps", 3)),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
        )
        child.session["memory"]["task"] = task
        child.session["memory"]["notes"] = [clip(self.history_text(), 300)]
        return "delegate_result:\n" + child.ask(task)


#############################################
#### Multi-Agent Supervisor Orchestrator ####
#############################################

