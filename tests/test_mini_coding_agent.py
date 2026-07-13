import json
import pytest
import httpx
from unittest.mock import patch

from mini_coding_agent import (
    FakeModelClient,
    MiniAgent,
    MultiAgentRunner,
    OllamaModelClient,
    OpenAiCompatibleClient,
    SessionStore,
    WorkspaceContext,
    build_supervisor_graph,
    build_welcome,
)


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    return MiniAgent(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_agent_runs_tool_then_final(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Read the file successfully.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Read the file successfully."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    assert "hello.txt" in agent.session["memory"]["files"]


def test_agent_retries_after_empty_model_output(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "<final>Recovered after retry.</final>",
        ],
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after retry."
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("empty response" in item for item in notices)


def test_agent_retries_after_malformed_tool_payload(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":"bad"}</tool>',
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            "<final>Recovered after malformed tool output.</final>",
        ],
    )

    answer = agent.ask("Inspect hello.txt")

    assert answer == "Recovered after malformed tool output."
    assert any(item["role"] == "tool" and item["name"] == "read_file" for item in agent.session["history"])
    notices = [item["content"] for item in agent.session["history"] if item["role"] == "assistant"]
    assert any("valid <tool> call" in item for item in notices)


def test_agent_accepts_xml_write_file_tool(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="hello.py"><content>print("hi")\n</content></tool>',
            "<final>Done.</final>",
        ],
    )

    answer = agent.ask("Create hello.py")

    assert answer == "Done."
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == 'print("hi")\n'


def test_retries_do_not_consume_the_whole_budget(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "",
            "<final>Recovered after several retries.</final>",
        ],
        max_steps=1,
    )

    answer = agent.ask("Do the task")

    assert answer == "Recovered after several retries."


def test_agent_saves_and_resumes_session(tmp_path):
    agent = build_agent(tmp_path, ["<final>First pass.</final>"])
    assert agent.ask("Start a session") == "First pass."

    resumed = MiniAgent.from_session(
        model_client=FakeModelClient(["<final>Resumed.</final>"]),
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )

    assert resumed.session["history"][0]["content"] == "Start a session"
    assert resumed.ask("Continue") == "Resumed."


def test_delegate_uses_child_agent(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"task":"inspect README","max_steps":2}}</tool>',
            "<final>Child result.</final>",
            "<final>Parent incorporated the child result.</final>",
        ],
    )

    answer = agent.ask("Use delegation")

    assert answer == "Parent incorporated the child result."
    tool_events = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert tool_events[0]["name"] == "delegate"
    assert "delegate_result" in tool_events[0]["content"]


def test_patch_file_replaces_exact_match(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "patch_file",
        {
            "path": "sample.txt",
            "old_text": "world",
            "new_text": "agent",
        },
    )

    assert result == "patched sample.txt"
    assert file_path.read_text(encoding="utf-8") == "hello agent\n"


def test_invalid_risky_tool_does_not_prompt_for_approval(tmp_path):
    agent = build_agent(tmp_path, [], approval_policy="ask")

    with patch("builtins.input") as mock_input:
        result = agent.run_tool("write_file", {})

    assert result.startswith("error: invalid arguments for write_file: 'path'")
    assert 'example: <tool name="write_file"' in result
    mock_input.assert_not_called()


def test_list_files_hides_internal_agent_state(tmp_path):
    agent = build_agent(tmp_path, [])
    (tmp_path / ".mini-coding-agent").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "hello.txt").write_text("hi\n", encoding="utf-8")

    result = agent.run_tool("list_files", {})

    assert ".mini-coding-agent" not in result
    assert ".git" not in result
    assert "[F] hello.txt" in result


def test_path_rejects_parent_escape(tmp_path):
    agent = build_agent(tmp_path, [])

    with pytest.raises(ValueError, match="path escapes workspace"):
        agent.path("../outside.txt")


def test_path_rejects_symlink_escape(tmp_path):
    agent = build_agent(tmp_path, [])
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    with pytest.raises(ValueError, match="path escapes workspace"):
        agent.path("outside-link/secret.txt")


def test_path_accepts_case_variant_on_case_insensitive_filesystems(tmp_path):
    project_root = tmp_path / "Proj"
    project_root.mkdir()
    agent = build_agent(project_root, [])
    variant = project_root.parent / project_root.name.lower() / "README.md"

    if not variant.exists():
        pytest.skip("case-sensitive filesystem")

    resolved = agent.path(str(variant))

    assert resolved.samefile(project_root / "README.md")


def test_repeated_identical_tool_call_is_rejected(tmp_path):
    agent = build_agent(tmp_path, [])
    agent.record({"role": "tool", "name": "run_shell", "args": {"command": "echo hi"}, "content": "hi", "created_at": "1"})
    agent.record({"role": "tool", "name": "run_shell", "args": {"command": "echo hi"}, "content": "hi", "created_at": "2"})

    result = agent.run_tool("run_shell", {"command": "echo hi"})

    assert result == "error: repeated identical tool call for run_shell; choose a different tool or return a final answer"


def test_repeated_read_only_tool_call_is_allowed(tmp_path):
    """search may repeat because small models often forget. list_files is limited to 2."""
    agent = build_agent(tmp_path, [])
    # list_files: 2 already in history, 3rd call should be rejected.
    for i in range(2):
        agent.record({"role": "tool", "name": "list_files", "args": {"path": "."}, "content": "(empty)", "created_at": str(i)})

    result = agent.run_tool("list_files", {"path": "."})
    assert "error: repeated identical tool call" in result

    # search is still unrestricted.
    for i in range(5):
        agent.record({"role": "tool", "name": "search", "args": {"pattern": "x"}, "content": "(empty)", "created_at": str(i)})
    assert "error: repeated identical tool call" not in agent.run_tool("search", {"pattern": "x"})


def test_read_file_repeat_allowed_up_to_three_times(tmp_path):
    """read_file for the same path is allowed up to 3 times, then rejected."""
    agent = build_agent(tmp_path, [])
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")

    # Seed history so repeated_tool_call sees prior reads.
    for i in range(3):
        agent.record({"role": "tool", "name": "read_file", "args": {"path": "a.txt"}, "content": "hello", "created_at": str(i)})

    # 4th read should be rejected.
    assert agent.repeated_tool_call("read_file", {"path": "a.txt"})
    # A different file is still fine.
    assert not agent.repeated_tool_call("read_file", {"path": "b.txt"})


def test_read_file_repeat_resets_after_write(tmp_path):
    """A write_file between reads resets the read_file repeat counter."""
    agent = build_agent(tmp_path, [])
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")

    # 3 reads in a row – next one should be rejected.
    for i in range(3):
        agent.record({"role": "tool", "name": "read_file", "args": {"path": "a.txt"}, "content": "hello", "created_at": str(i)})
    assert agent.repeated_tool_call("read_file", {"path": "a.txt"})

    # After a write, counter resets.
    agent.record({"role": "tool", "name": "write_file", "args": {"path": "a.txt", "content": "world\n"}, "content": "wrote", "created_at": "4"})
    assert not agent.repeated_tool_call("read_file", {"path": "a.txt"})


def test_list_files_allowed_up_to_two_times(tmp_path):
    """list_files is allowed up to 2 times for the same directory, then rejected."""
    agent = build_agent(tmp_path, [])
    for i in range(2):
        agent.record({"role": "tool", "name": "list_files", "args": {"path": "."}, "content": "(empty)", "created_at": str(i)})
    # 3rd call should be rejected.
    assert agent.repeated_tool_call("list_files", {"path": "."})
    # Different directory is still fine.
    assert not agent.repeated_tool_call("list_files", {"path": "tests"})


def test_welcome_screen_keeps_box_shape_for_long_paths(tmp_path):
    deep = tmp_path / "very" / "long" / "path" / "for" / "the" / "mini" / "agent" / "welcome" / "screen"
    deep.mkdir(parents=True)
    agent = build_agent(deep, [])

    welcome = build_welcome(agent, model="qwen3.5:4b", backend="Ollama:http://127.0.0.1:11434")
    lines = welcome.splitlines()

    assert len(lines) >= 5
    assert len({len(line) for line in lines}) == 1
    assert "..." in welcome
    assert "O   O" in welcome
    assert "MINI-CODING-AGENT" not in welcome
    assert "MINI CODING AGENT" in welcome
    assert "// READY" not in welcome
    assert "SLASH" not in welcome
    assert "READY      " not in welcome
    assert "commands: Commands:" not in welcome


def test_prompt_top_level_sections_stay_flush_left_with_multiline_content(tmp_path):
    workspace = WorkspaceContext(
        cwd=str(tmp_path),
        repo_root=str(tmp_path),
        branch="fix/prompt-indentation",
        default_branch="main",
        status=" M mini_coding_agent.py\n?? tests/test_prompt.py",
        recent_commits=["abc123 first commit", "def456 second commit"],
        project_docs={"README.md": "line1\nline2"},
    )
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    agent = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )
    agent.session["memory"] = {
        "task": "verify prompt formatting",
        "files": ["mini_coding_agent.py"],
        "notes": ["saw inconsistent indentation", "need regression coverage"],
    }
    agent.record({"role": "user", "content": "inspect prompt()", "created_at": "1"})
    agent.record(
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "mini_coding_agent.py"},
            "content": "    def prompt(self, user_message):\n        ...",
            "created_at": "2",
        }
    )

    prompt = agent.prompt("is this issue legit?")
    lines = prompt.splitlines()

    for label in ["Rules:", "Tools:", "Valid response examples:", "Workspace:", "Memory:", "Transcript:", "Current user request:"]:
        assert label in lines
        assert f"            {label}" not in prompt


def _make_filler(i):
    return {"role": "tool", "name": "list_files", "args": {}, "content": "", "created_at": str(i)}


def test_history_text_deduplicates_reads_but_not_after_write(tmp_path):
    """read_file deduplication must not skip a read that follows a write.

    Realistic prior-turn history (non-recent window):
        user: "update config"
        assistant: <tool>read_file config</tool>
        tool:   config v1 (content: setting=true)
        assistant: <tool>write_file config</tool>
        tool:   wrote
        assistant: <tool>read_file config</tool>
        tool:   config v2 (content: setting=false)   <- MUST NOT be skipped

    Without fix: seen_reads={"config"} after first read; write does NOT clear it;
                 second read is wrongly skipped (LLM sees stale content).
    With fix: write clears seen_reads, second read is correctly shown.
    """
    agent = build_agent(tmp_path, [])

    # Simulate a prior turn with read->write->read on the same file
    # history_length=13, recent_start=7 (indices 0-6 non-recent, 7-12 recent)
    agent.record({"role": "user", "content": "update config", "created_at": "0"})        # index 0
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"config.txt"}}</tool>', "created_at": "1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"}, "content": "# config.txt\n   1: setting=true\n", "created_at": "2"})  # index 2, non-recent, ADDED
    agent.record({"role": "assistant", "content": '<tool>{"name":"write_file","args":{"path":"config.txt","content":"setting=false\n"}}</tool>', "created_at": "3"})
    agent.record({"role": "tool", "name": "write_file", "args": {"path": "config.txt", "content": "setting=false\n"}, "content": "wrote config.txt", "created_at": "4"})  # index 4, non-recent
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"config.txt"}}</tool>', "created_at": "5"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "config.txt"}, "content": "# config.txt\n   1: setting=false\n", "created_at": "6"})  # index 6, non-recent, ADDED (write cleared dedup)
    # recent entries
    for i in range(7, 13):
        agent.record(_make_filler(i))

    history = agent.history_text()

    # Both read contents appear exactly once (check full line to avoid JSON false positives)
    assert "# config.txt\n   1: setting=true\n" in history
    assert "# config.txt\n   1: setting=false\n" in history
    # Also verify duplicate read (setting=true, same path) does NOT appear twice
    assert history.count("setting=true") == 1


def test_history_text_deduplicates_unchanged_repeated_reads(tmp_path):
    """read_file deduplication should still skip repeated reads with no write in between."""
    agent = build_agent(tmp_path, [])

    # Realistic: two identical reads with no write between them
    # history_length=10, recent_start=4 (indices 0-3 non-recent, 4-9 recent)
    agent.record({"role": "user", "content": "check logs", "created_at": "0"})  # index 0
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"log.txt"}}</tool>', "created_at": "1"})
    agent.record({"role": "tool", "name": "read_file", "args": {"path": "log.txt"}, "content": "# log.txt\n   1: stable\n", "created_at": "2"})  # index 2, non-recent, ADDED
    agent.record({"role": "assistant", "content": '<tool>{"name":"read_file","args":{"path":"log.txt"}}</tool>', "created_at": "3"})  # index 3, non-recent, SKIPPED (dup)
    for i in range(4, 10):
        agent.record(_make_filler(i))  # indices 4-9, recent

    history = agent.history_text()

    # Only first read should appear; duplicates must be skipped
    assert history.count("stable") == 1


def test_ollama_client_posts_expected_payload():
    captured = {}

    def fake_handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"response": "<final>ok</final>"})

    client = OllamaModelClient(
        model="qwen3.5:4b",
        host="http://127.0.0.1:11434",
        temperature=0.2,
        top_p=0.9,
        timeout=30,
    )

    original_client = httpx.Client
    try:
        httpx.Client = lambda **kwargs: original_client(transport=httpx.MockTransport(fake_handler), **kwargs)
        result = client.complete("hello", 42)
    finally:
        httpx.Client = original_client

    assert result == "<final>ok</final>"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["method"] == "POST"
    assert captured["body"]["model"] == "qwen3.5:4b"
    assert captured["body"]["prompt"] == "hello"
    assert captured["body"]["stream"] is False
    assert captured["body"]["raw"] is False
    assert captured["body"]["think"] is False
    assert captured["body"]["options"]["num_predict"] == 42


#############################################
#### Multi-Agent Tests ######################
#############################################

from mini_coding_agent import MultiAgentRunner


def test_planner_role_has_no_write_tools(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    planner = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="planner",
    )

    assert "write_file" not in planner.tools
    assert "patch_file" not in planner.tools
    assert "list_files" in planner.tools
    assert "read_file" in planner.tools
    assert "search" in planner.tools


def test_planner_role_prompt_contains_architect_instructions(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    planner = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="planner",
    )

    prefix = planner.build_prefix()
    assert "Software Architect Planner" in prefix
    assert "Do NOT write code" in prefix


def test_coder_role_prompt_contains_coder_instructions(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    coder = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="coder",
    )

    prefix = coder.build_prefix()
    assert "Software Engineer" in prefix
    assert "write_file" in coder.tools
    assert "patch_file" in coder.tools


def test_multi_agent_runner_planner_then_coder(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")

    planner = MiniAgent(
        model_client=FakeModelClient([
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            "<final>Plan: Read hello.txt and summarize contents.</final>",
        ]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="planner",
    )
    coder = MiniAgent(
        model_client=FakeModelClient([
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":2}}</tool>',
            "<final>Summary: alpha and beta.</final>",
        ]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="coder",
    )
    tester = MiniAgent(
        model_client=FakeModelClient(["<final>Tests passed. No issues.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="tester",
    )
    reviewer = MiniAgent(
        model_client=FakeModelClient(["<final>approved</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="reviewer",
    )

    runner = MultiAgentRunner(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
    )
    runner.planner = planner
    runner.coder = coder
    runner.tester = tester
    runner.reviewer = reviewer
    runner.graph = build_supervisor_graph(planner, coder, tester, reviewer)

    result = runner.ask("Inspect hello.txt")

    assert result == "Summary: alpha and beta."
    assert "Plan: Read hello.txt" in runner.planner.session["memory"]["task"] or any(
        "Plan: Read hello.txt" in str(item.get("content", "")) for item in runner.planner.session["history"]
    )
    assert any(
        item["role"] == "tool" and item["name"] == "read_file"
        for item in runner.coder.session["history"]
    )


def test_tester_role_has_write_tools_when_not_read_only(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    tester = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="tester",
        read_only=False,
    )

    assert "write_file" in tester.tools
    assert "patch_file" in tester.tools
    assert "list_files" in tester.tools
    assert "read_file" in tester.tools


def test_tester_role_has_no_write_tools_when_read_only(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    tester = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="tester",
        read_only=True,
    )

    assert "write_file" not in tester.tools
    assert "patch_file" not in tester.tools
    assert "list_files" in tester.tools
    assert "read_file" in tester.tools


def test_tester_role_prompt_contains_tester_instructions(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    tester = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="tester",
        read_only=False,
    )

    prefix = tester.build_prefix()
    assert "Quality Assurance Tester" in prefix
    assert "Do NOT run the same pytest command repeatedly" in prefix
    assert "write_file" in prefix


def test_reviewer_role_has_no_write_tools(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    reviewer = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="reviewer",
    )

    assert "write_file" not in reviewer.tools
    assert "patch_file" not in reviewer.tools
    assert "read_file" in reviewer.tools


def test_reviewer_role_prompt_contains_reviewer_instructions(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    reviewer = MiniAgent(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="reviewer",
    )

    prefix = reviewer.build_prefix()
    assert "Senior Code Reviewer" in prefix
    assert "approved" in prefix
    assert "needs_fix" in prefix


def test_multi_agent_full_pipeline_approved(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")

    planner = MiniAgent(
        model_client=FakeModelClient(["<final>Plan: create hello.py.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="planner",
    )
    coder = MiniAgent(
        model_client=FakeModelClient(["<final>Created hello.py.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="coder",
    )
    tester = MiniAgent(
        model_client=FakeModelClient(["<final>Tests passed.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="tester",
    )
    reviewer = MiniAgent(
        model_client=FakeModelClient(["<final>approved</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="reviewer",
    )

    runner = MultiAgentRunner(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
    )
    runner.planner = planner
    runner.coder = coder
    runner.tester = tester
    runner.reviewer = reviewer
    runner.graph = build_supervisor_graph(planner, coder, tester, reviewer)

    result = runner.ask("Create hello.py")

    assert result == "Created hello.py."
    assert runner.reviewer.session["history"]


def test_multi_agent_needs_fix_then_approved(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")

    planner = MiniAgent(
        model_client=FakeModelClient(["<final>Plan: fix greeting.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="planner",
    )
    coder = MiniAgent(
        model_client=FakeModelClient([
            "<final>First draft.</final>",
            "<final>Fixed draft.</final>",
        ]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="coder",
    )
    tester = MiniAgent(
        model_client=FakeModelClient([
            "<final>Test report 1.</final>",
            "<final>Test report 2.</final>",
        ]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="tester",
    )
    reviewer = MiniAgent(
        model_client=FakeModelClient([
            "<final>needs_fix: missing docstring</final>",
            "<final>approved</final>",
        ]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="reviewer",
    )

    runner = MultiAgentRunner(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
    )
    runner.planner = planner
    runner.coder = coder
    runner.tester = tester
    runner.reviewer = reviewer
    runner.graph = build_supervisor_graph(planner, coder, tester, reviewer)

    result = runner.ask("Fix greeting")

    assert result == "Fixed draft."
    assert len([item for item in runner.coder.session["history"] if item["role"] == "assistant"]) == 2


def test_multi_agent_max_cycles(tmp_path):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")

    planner = MiniAgent(
        model_client=FakeModelClient(["<final>Plan.</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="planner",
    )
    coder = MiniAgent(
        model_client=FakeModelClient([
            "<final>Draft 1.</final>",
            "<final>Draft 2.</final>",
            "<final>Draft 3.</final>",
        ]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="coder",
    )
    tester = MiniAgent(
        model_client=FakeModelClient([
            "<final>Fail.</final>",
            "<final>Fail.</final>",
            "<final>Fail.</final>",
        ]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="tester",
    )
    reviewer = MiniAgent(
        model_client=FakeModelClient([
            "<final>needs_fix: bug 1</final>",
            "<final>needs_fix: bug 2</final>",
            "<final>needs_fix: bug 3</final>",
        ]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        role="reviewer",
    )

    runner = MultiAgentRunner(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
    )
    runner.planner = planner
    runner.coder = coder
    runner.tester = tester
    runner.reviewer = reviewer
    runner.graph = build_supervisor_graph(planner, coder, tester, reviewer)

    result = runner.ask("Task")

    assert result == "Draft 3."
    assert len([item for item in runner.coder.session["history"] if item["role"] == "assistant"]) == 3


def test_parse_reviewer_verdict_approved():
    from mini_coding_agent import _parse_reviewer_verdict
    verdict, feedback = _parse_reviewer_verdict("approved")
    assert verdict == "approved"
    assert feedback == "approved"


def test_parse_reviewer_verdict_needs_fix():
    from mini_coding_agent import _parse_reviewer_verdict
    verdict, feedback = _parse_reviewer_verdict("needs_fix: missing docstring")
    assert verdict == "needs_fix"
    assert feedback == "missing docstring"


def test_parse_reviewer_verdict_fallback():
    from mini_coding_agent import _parse_reviewer_verdict
    verdict, feedback = _parse_reviewer_verdict("something else")
    assert verdict == "needs_fix"
    assert feedback == "something else"


def test_openai_compatible_client_posts_expected_payload():
    captured = {}

    def fake_handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "<final>ok</final>"}}]
        })

    client = OpenAiCompatibleClient(
        model="gpt-4o-mini",
        base_url="https://api.example.com/v1",
        api_key="sk-test123",
        temperature=0.2,
        top_p=0.9,
        timeout=30,
    )

    original_client = httpx.Client
    try:
        httpx.Client = lambda **kwargs: original_client(transport=httpx.MockTransport(fake_handler), **kwargs)
        result = client.complete("hello", 42)
    finally:
        httpx.Client = original_client

    assert result == "<final>ok</final>"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["body"]["max_tokens"] == 42
    assert captured["headers"]["authorization"] == "Bearer sk-test123"
    assert captured["headers"]["content-type"] == "application/json"
