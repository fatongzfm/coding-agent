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
    IGNORED_FILE_NAMES,
    MAX_HISTORY,
    MAX_TOOL_OUTPUT,
)
from mini_coding_agent.observability import make_emitter, metrics_collector
from mini_coding_agent.skills.base import SkillRegistry

# Ensure built-in skills are registered at import time
SkillRegistry.discover_builtin()

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
        agent.emit(agent.role, "llm_call", {"prompt_chars": len(prompt), "max_tokens": agent.max_new_tokens})
        t0 = time.perf_counter()
        raw = agent.model_client.complete(prompt, agent.max_new_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000
        metrics_collector.record_llm_call(
            agent.run_id or agent.session["id"],
            latency_ms,
            len(prompt),
            len(raw),
        )
        logger.debug("model_response role=%s response_chars=%d", agent.role, len(raw))
        agent.emit(agent.role, "llm_output", {"raw": raw, "response_chars": len(raw)})
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
        metrics_collector.record_tool_call(
            agent.run_id or agent.session["id"],
            duration_ms,
        )
        logger.info(
            "tool_execution tool=%s duration_ms=%d result_chars=%d",
            name, duration_ms, len(result),
        )
        agent.emit(agent.role, "tool_result", {"tool": name, "args": args, "result": result, "duration_ms": duration_ms})
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
        agent.emit(agent.role, "retry", {"reason": str(payload)})
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
        agent.emit(agent.role, "final_answer", {"answer": final})
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
        self.run_id: str | None = None
        self.tools = self.build_tools()
        self.prefix = self.build_prefix()
        self.session_path = self.session_store.save(self.session)
        self.graph = _build_agent_graph(self)

    def emit(self, node: str, event_type: str, payload: dict | None = None):
        if self.run_id is None:
            return
        emitter = make_emitter(self.run_id)
        emitter(node, event_type, payload)

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
        # Build tools from the skill registry with role-based filtering
        skills = SkillRegistry.list_skills()
        tools = {}
        for name, skill in skills.items():
            # Filter write/patch permissions based on role and read_only mode
            if name in ("write_file", "patch_file"):
                if self.read_only or (self.role == "tester" and self.read_only):
                    continue
                if self.role not in ("coder", "tester"):
                    continue
            # Filter delegate based on depth
            if name == "delegate" and self.depth >= self.max_depth:
                continue
            tools[name] = {
                "schema": skill.get_schema_dict(),
                "risky": skill.risky,
                "description": skill.description,
                "run": lambda args, s=skill: s.validate_and_run(self, args),
            }
        return tools

    def build_tools_legacy(self):
        """Legacy hard-coded tool builder (kept for reference / fallback)."""
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
        if self.role == "coder" or (self.role == "tester" and not self.read_only):
            tools["write_file"] = {
                "schema": {"path": "str", "content": "str"},
                "risky": True,
                "description": 'Write a text file. MUST use XML: <tool name="write_file" path="file.py"><content>...</content></tool>',
                "run": self.tool_write_file,
            }
            tools["patch_file"] = {
                "schema": {"path": "str", "old_text": "str", "new_text": "str"},
                "risky": True,
                "description": 'Replace one exact text block in a file. MUST use XML: <tool name="patch_file" path="file.py"><old_text>...</old_text><new_text>...</new_text></tool>',
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
        # Build tool list: read-only agents don't need write_file / patch_file
        is_read_only = self.read_only
        tool_lines = []
        for name, tool in self.tools.items():
            if is_read_only and name in ("write_file", "patch_file"):
                continue
            schema = tool["schema"]
            # Handle both legacy string dicts and new Pydantic JSON Schema
            if isinstance(schema, dict) and "properties" in schema:
                field_parts = []
                for key, prop in schema.get("properties", {}).items():
                    ptype = prop.get("type", "any")
                    if "default" in prop:
                        field_parts.append(f"{key}: {ptype}={prop['default']}")
                    elif key not in schema.get("required", []):
                        field_parts.append(f"{key}: {ptype} (optional)")
                    else:
                        field_parts.append(f"{key}: {ptype}")
                fields = ", ".join(field_parts)
            else:
                fields = ", ".join(f"{key}: {value}" for key, value in schema.items())
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        tool_text = "\n".join(tool_lines)

        # Examples: read-only roles don't need write_file / patch_file XML examples
        if is_read_only:
            examples = "\n".join(
                [
                    '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                    '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                    '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                    "<final>Done.</final>",
                ]
            )
        else:
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

        # Rules: read-only roles don't need XML file-writing rules
        common_rules = [
            "- Use tools instead of guessing about the workspace.",
            "- Return exactly ONE <tool>...</tool> or ONE <final>...</final>. Multiple tools in a single response are NOT allowed.",
            "- Tool calls must look like:",
            '  <tool>{"name":"tool_name","args":{...}}</tool>',
        ]
        write_rules = [
            "- For write_file and patch_file you MUST use XML style. JSON is NOT allowed for these tools:",
            '  <tool name="write_file" path="file.py"><content>...</content></tool>',
            '  <tool name="patch_file" path="file.py"><old_text>...</old_text><new_text>...</new_text></tool>',
        ]
        final_rules = [
            "- Final answers must look like:",
            "  <final>your answer</final>",
            "- Never invent tool results.",
            "- Keep answers concise and concrete.",
        ]
        coder_rules = [
            "- If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.",
            "- Before writing tests for existing code, read the implementation first.",
            "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
            "- New files should be complete and runnable, including obvious imports.",
            "- If tests fail, read the error output carefully, fix the code, then re-run tests. Do NOT repeat the same test command without fixing the code first.",
        ]
        if is_read_only:
            rules = "\n".join(common_rules + final_rules + [
                "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.",
                "- Required tool arguments must not be empty. Do not call read_file, run_shell, or delegate with args={}.",
            ])
        else:
            rules = "\n".join(common_rules + write_rules + final_rules + coder_rules + [
                "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.",
                "- Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={}.",
            ])
        # Load role identity from config if available, else fallback to defaults
        from mini_coding_agent.config import load_config
        cfg = load_config(None)
        roles_cfg = cfg.get("roles", {})
        role_cfg = roles_cfg.get(self.role, {})
        if role_cfg.get("identity"):
            identity = role_cfg["identity"]
        elif self.role == "planner":
            identity = (
                "You are a Software Architect Planner, a specialized agent that analyzes "
                "requirements and produces implementation plans. Do NOT write code. "
                "Only produce a concise plan (under 150 words).\n\n"
                "EFFICIENCY RULES:\n"
                "- You have a LIMITED step budget. Do NOT waste steps reading large unrelated files.\n"
                "- Do NOT read the project's own test files (e.g. tests/test_mini_coding_agent.py) "
                "unless the task explicitly asks about them.\n"
                "- Use search FIRST to check if the requested functionality already exists, "
                "before listing directories or reading files.\n"
                "- If the task is straightforward (e.g. implementing a single algorithm), "
                "output the plan in 1-2 steps without deep exploration.\n"
                "- Do NOT read the same file more than once.\n"
                "- If you see the code you need already exists, mention it in the plan and stop.\n\n"
                "ROLE ASSIGNMENT:\n"
                "- Your plan should ONLY contain implementation steps for the [coder].\n"
                "- Do NOT include testing steps, test files, or any [tester] assignments in your plan.\n"
                "- The [coder] only writes implementation source files (e.g. src/, lib/, module code).\n"
                "- Testing will be handled later by the tester after the coder finishes."
            )
        elif self.role == "tester":
            identity = (
                "You are a Quality Assurance Tester. The coder has written the implementation "
                "but did NOT write tests. Your job is to write tests, run them, and report the results.\n\n"
                "EFFICIENCY RULES:\n"
                "- You have a VERY LIMITED step budget. Read each file AT MOST ONCE.\n"
                "- If read_file returns 'repeated identical tool call', that means you already have the file content. "
                "Do NOT try to read it again — use what you already know.\n"
                "- If list_files returns 'repeated identical tool call', that means you already listed this directory. "
                "Do NOT try to list it again.\n"
                "- If all tests already pass and coverage is adequate, return a final answer immediately. "
                "Do NOT run the same pytest command repeatedly.\n"
                "- Write focused tests, run them, and return a concise pass/fail report with any issues found."
            )
        elif self.role == "reviewer":
            identity = (
                "You are a Senior Code Reviewer, a specialized read-only agent that performs "
                "code review, security checks, and best-practice validation. Do NOT modify any files.\n\n"
                "EFFICIENCY RULES:\n"
                "- You have a VERY LIMITED step budget (only a few steps). Be EXTREMELY efficient.\n"
                "- If the coder has already reported that tests PASSED, and the implementation looks correct, "
                "you may directly return <final>approved</final> without re-reading files.\n"
                "- If read_file returns 'repeated identical tool call', that means the file has already been read. "
                "Do NOT try to read it again — use what you already know.\n"
                "- Read each file AT MOST ONCE.\n"
                "Return your verdict as either:\n"
                "  <final>approved</final>\n"
                "or\n"
                "  <final>needs_fix: [specific, actionable feedback]</final>"
            )
        else:
            identity = (
                "You are an expert Software Engineer. Your job is to implement the given plan "
                "by writing complete, runnable code. You MUST follow the plan exactly.\n\n"
                "EFFICIENCY RULES:\n"
                "- You have a LIMITED step budget. Focus on writing code, not exploring files.\n"
                "- Do NOT read the same file more than once. If you need to reference it again, use what you already know.\n"
                "- Do NOT write tests — the tester will write them. Only write the implementation code.\n"
                "- If read_file returns 'repeated identical tool call', that means you already have the file content. "
                "Do NOT try to read it again — use what you already know.\n"
                "- If the plan is clear, start writing code immediately without excessive file exploration.\n\n"
                "OUTPUT RULES:\n"
                "- After completing the implementation, your final answer MUST include a 'Test Plan' section "
                "describing what tests the tester should write (e.g., edge cases, error inputs, boundary conditions, "
                "and expected behaviors).\n"
                "- Do NOT run tests yourself unless they are broken and need fixing."
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
        recent_start = max(0, len(history) - 10)
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
                limit = 900 if recent else 100
                # P1: aggressively compress large read_file results in history
                if item["name"] == "read_file" and not recent and len(item.get("content", "")) > 1500:
                    limit = 60
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 120
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
    def _read_file_preview(self, path: str, limit: int = 100) -> str:
        """Return a short preview of a file for use in error messages."""
        try:
            resolved = self.path(path)
            if not resolved.is_file():
                return ""
            text = resolved.read_text(encoding="utf-8", errors="replace")
            preview = text[:limit].replace("\n", " ")
            if len(text) > limit:
                preview += " ..."
            return preview
        except Exception:
            return ""

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
            if name == "read_file":
                preview = self._read_file_preview(args.get("path", ""))
                return (
                    f"error: repeated identical tool call for {name}. "
                    f"You have already read this file. Content preview: {preview}\n"
                    f"Choose a different tool or return a final answer."
                )
            if name == "list_files":
                return (
                    f"error: repeated identical tool call for {name}. "
                    f"You have already listed this directory. "
                    f"Choose a different tool or return a final answer."
                )
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        if tool["risky"] and not self.approve(name, args):
            return f"error: approval denied for {name}"
        try:
            return clip(tool["run"](args))
        except Exception as exc:
            return f"error: tool {name} failed: {exc}"

    def repeated_tool_call(self, name, args):
        """Reject only truly pointless repeats.

        A repeat is allowed if the agent has written or patched a file since the
        previous identical call — the workspace state may have changed.
        search is always allowed to repeat because small models often "forget".
        list_files is allowed up to 2 times for the same directory.
        read_file is allowed up to 3 times for the same file before being
        rejected, to prevent step exhaustion from repetitive re-reading.
        """
        if name == "search":
            return False
        if name == "list_files":
            path = str(args.get("path", "."))
            count = 0
            for item in reversed(self.session["history"]):
                if (
                    item.get("role") == "tool"
                    and item.get("name") == "list_files"
                    and str(item.get("args", {}).get("path", ".")) == path
                ):
                    count += 1
            return count >= 2
        if name == "read_file":
            path = str(args.get("path", ""))
            if not path:
                return False
            # Count how many times this file has been read since the last write.
            count = 0
            for item in reversed(self.session["history"]):
                if item.get("role") == "tool" and item.get("name") in ("write_file", "patch_file"):
                    break
                if (
                    item.get("role") == "tool"
                    and item.get("name") == "read_file"
                    and str(item.get("args", {}).get("path", "")) == path
                ):
                    count += 1
            return count >= 3
        history = self.session["history"]
        if len(history) < 2:
            return False

        # Walk backwards by index to find the previous identical tool call.
        for i in range(len(history) - 2, -1, -1):
            item = history[i]
            if item["role"] == "tool" and item["name"] == name and item["args"] == args:
                # If any write/patch happened between then and now, the repeat is justified.
                for mid in history[i + 1 :]:
                    if mid["role"] == "tool" and mid["name"] in ("write_file", "patch_file"):
                        return False
                return True
            # Stop searching once we hit a *different* tool call of the same name.
            if item["role"] == "tool" and item["name"] == name:
                break
        return False

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

        # Phase 1: Pydantic schema validation (if skill is registered)
        skill = SkillRegistry.get(name)
        if skill is not None and skill.param_model is not None:
            from pydantic import ValidationError

            try:
                validated = skill.param_model.model_validate(args)
                # Merge validated defaults back into args for legacy checks below
                args = validated.model_dump()
            except ValidationError as exc:
                # Produce a concise single-field error message for backward compat
                first_error = exc.errors()[0]
                field = first_error.get("loc", [""])[0] if first_error.get("loc") else ""
                raise ValueError(f"'{field}'") from exc
            except Exception as exc:
                raise ValueError(str(exc)) from exc

        # Phase 2: Legacy runtime / filesystem validation (kept for backward compat)
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
        # Tester needs run_shell to execute tests; it is still read-only for file writes.
        if self.read_only and name == "run_shell":
            return True
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
        # Some models output multiple <tool> blocks at once; keep only the first one.
        if raw.count("<tool>") > 1 or raw.count("<tool ") > 1:
            first_tag = raw.find("<tool>")
            if first_tag == -1:
                first_tag = raw.find("<tool ")
            close_tag = raw.find("</tool>", first_tag)
            if close_tag != -1:
                raw = raw[first_tag : close_tag + len("</tool>")]
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = MiniAgent.extract(raw, "tool")
            payload = None
            try:
                payload = json.loads(body)
            except Exception:
                # Tolerance: try to recover flat JSON for write_file / patch_file
                payload = MiniAgent._recover_flat_json(body)
                if payload is None:
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
            # Tolerance: promote top-level keys to args for write_file / patch_file
            name = str(payload.get("name", "")).strip()
            if name in ("write_file", "patch_file") and not args:
                args = {k: v for k, v in payload.items() if k not in ("name", "args")}
                payload["args"] = args
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
    def _recover_flat_json(body: str) -> dict | None:
        """Best-effort recovery for models that put fields at top-level instead of inside args."""
        # Try to extract name, path, content, old_text, new_text with regex
        name_match = re.search(r'"name"\s*:\s*"([^"]*)"', body)
        if not name_match:
            return None
        name = name_match.group(1)
        if name not in ("write_file", "patch_file"):
            return None
        result = {"name": name, "args": {}}
        for key in ("path", "content", "old_text", "new_text"):
            pattern = rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"'
            m = re.search(pattern, body)
            if m:
                # Unescape JSON string
                raw_val = m.group(1)
                try:
                    result["args"][key] = json.loads(f'"{raw_val}"')
                except Exception:
                    result["args"][key] = raw_val
        if not result["args"]:
            return None
        return result

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For write_file and patch_file you MUST use XML: <tool name="write_file" path="file.py"><content>...</content></tool>.'
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

    def _is_ignored_file(self, path: Path) -> bool:
        return path.name in IGNORED_FILE_NAMES

    def tool_list_files(self, args):
        path = self.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        entries = [
            item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
            if item.name not in IGNORED_PATH_NAMES and not self._is_ignored_file(item)
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
        if self._is_ignored_file(path):
            return (
                f"# {path.relative_to(self.root)}\n"
                "   1: (This file is part of the agent's own test suite and is irrelevant to the current task.)"
            )
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
            lines = result.stdout.strip().splitlines()
            filtered = [
                line for line in lines
                if not any(name in line for name in IGNORED_FILE_NAMES)
            ]
            return "\n".join(filtered) or result.stderr.strip() or "(no matches)"

        matches = []
        files = [path] if path.is_file() else [
            item for item in path.rglob("*")
            if item.is_file()
            and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(self.root).parts)
            and not self._is_ignored_file(item)
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

