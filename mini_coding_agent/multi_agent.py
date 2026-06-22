import logging
import uuid
from typing import TypedDict

from mini_coding_agent.agent import MiniAgent
from mini_coding_agent.context import clip
from mini_coding_agent.observability import make_emitter

class SupervisorState(TypedDict):
    user_message: str
    plan: str | None
    test_report: str | None
    review_feedback: str | None
    review_verdict: str | None
    review_cycles: int
    final_answer: str | None


def _gather_context(*agents: MiniAgent) -> str:
    """Summarize upstream agents' memory (files inspected + observations)."""
    parts = []
    for agent in agents:
        memory = agent.session["memory"]
        if memory.get("files"):
            files = ", ".join(memory["files"])
            parts.append(f"- [{agent.role}] inspected files: {files}")
        if memory.get("notes"):
            for note in memory["notes"]:
                parts.append(f"- [{agent.role}] {note}")
    return clip("\n".join(parts), 1200) if parts else ""


def _parse_reviewer_verdict(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    lower = raw.lower()
    if lower.startswith("approved"):
        return "approved", raw
    if lower.startswith("needs_fix"):
        feedback = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return "needs_fix", feedback
    return "needs_fix", raw


def build_supervisor_graph(planner: MiniAgent, coder: MiniAgent, tester: MiniAgent, reviewer: MiniAgent, run_id: str | None = None, max_review_cycles: int = 3):
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(SupervisorState)
    emit = make_emitter(run_id)

    def planner_node(state: SupervisorState):
        emit("planner", "node_start")
        plan = planner.ask(state["user_message"])
        emit("planner", "node_end", {"plan": plan})
        return {"plan": plan}

    def _is_failed_plan(plan: str | None) -> bool:
        return plan is None or "Stopped after reaching" in plan or "step limit" in plan.lower()

    def coder_node(state: SupervisorState):
        emit("coder", "node_start", {"cycle": state.get("review_cycles", 0)})
        plan = state.get("plan")
        if _is_failed_plan(plan):
            # Planner failed — fall back to the original user request.
            parts = [f"Task: {state['user_message']}"]
            parts.append("(The planner did not produce a plan. Proceed based on the task above.)")
        else:
            parts = [f"Plan:\n{plan}"]

        upstream = _gather_context(planner)
        if upstream:
            parts.append(f"Context from previous stages:\n{upstream}")

        if state.get("review_feedback"):
            parts.append(f"Review feedback:\n{state['review_feedback']}")
        if not _is_failed_plan(plan):
            parts.append(f"Implement: {state['user_message']}")
        prompt = "\n\n".join(parts)
        result = coder.ask(prompt)
        emit("coder", "node_end", {"final_answer": result})
        return {"final_answer": result}

    def _is_step_limit_failure(text: str | None) -> bool:
        return text is not None and ("Stopped after reaching" in text or "step limit" in text.lower())

    def tester_node(state: SupervisorState):
        emit("tester", "node_start")
        if _is_step_limit_failure(state.get("final_answer")):
            report = "Coder failed due to step exhaustion — no code to test."
            emit("tester", "node_end", {"test_report": report})
            return {"test_report": report}

        upstream = _gather_context(planner, coder)
        context_section = f"\n\nContext from previous stages:\n{upstream}" if upstream else ""

        prompt = (
            f"Original request: {state['user_message']}\n\n"
            f"Implementation and Test Plan from coder:\n{state['final_answer']}{context_section}\n\n"
            "Your job is to write tests based on the Test Plan above, run them, and report the results. "
            "If tests already exist, review their coverage against the Test Plan, add any missing edge cases, then run them. "
            "Return a concise test report with pass/fail status and any issues found."
        )
        report = tester.ask(prompt)
        emit("tester", "node_end", {"test_report": report})
        return {"test_report": report}

    logger = logging.getLogger("mca.multi")

    def reviewer_node(state: SupervisorState):
        emit("reviewer", "node_start", {"cycle": state["review_cycles"] + 1})
        if _is_step_limit_failure(state.get("final_answer")):
            emit("reviewer", "node_end", {"verdict": "needs_fix", "feedback": "Coder failed due to step exhaustion."})
            return {
                "review_verdict": "needs_fix",
                "review_feedback": "Coder failed due to step exhaustion.",
                "review_cycles": state["review_cycles"] + 1,
            }
        if _is_step_limit_failure(state.get("test_report")):
            emit("reviewer", "node_end", {"verdict": "needs_fix", "feedback": "Tester failed due to step exhaustion."})
            return {
                "review_verdict": "needs_fix",
                "review_feedback": "Tester failed due to step exhaustion.",
                "review_cycles": state["review_cycles"] + 1,
            }

        upstream = _gather_context(planner, coder, tester)
        context_section = f"\n\nContext from previous stages:\n{upstream}" if upstream else ""

        prompt = (
            f"Original request: {state['user_message']}\n\n"
            f"Implementation:\n{state['final_answer']}{context_section}\n\n"
            f"Test report:\n{state['test_report']}\n\n"
            "Return <final>approved</final> or <final>needs_fix: [detailed feedback]</final>."
        )
        raw = reviewer.ask(prompt)
        verdict, feedback = _parse_reviewer_verdict(raw)
        logger.info("reviewer_verdict verdict=%s cycle=%d feedback_chars=%d", verdict, state["review_cycles"] + 1, len(feedback))
        emit("reviewer", "node_end", {"verdict": verdict, "feedback": feedback})
        return {
            "review_verdict": verdict,
            "review_feedback": feedback,
            "review_cycles": state["review_cycles"] + 1,
        }

    def supervisor_decision(state: SupervisorState):
        if state["review_verdict"] == "approved":
            logger.info("supervisor_decision approved cycle=%d", state["review_cycles"])
            return "approved"
        if state["review_cycles"] >= max_review_cycles:
            logger.info("supervisor_decision max_cycles reached=%d", state["review_cycles"])
            return "max_cycles"
        logger.info("supervisor_decision needs_fix cycle=%d", state["review_cycles"])
        return "needs_fix"

    builder.add_node("planner", planner_node)
    builder.add_node("coder", coder_node)
    builder.add_node("tester", tester_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "coder")
    builder.add_edge("coder", "tester")
    builder.add_edge("tester", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        supervisor_decision,
        {"approved": END, "max_cycles": END, "needs_fix": "coder"},
    )

    return builder.compile()


class MultiAgentRunner:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        approval_policy="ask",
        max_steps_planner=4,
        max_steps_coder=6,
        max_steps_tester=4,
        max_steps_reviewer=4,
        max_new_tokens=512,
        max_review_cycles=3,
    ):
        self.planner = MiniAgent(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            approval_policy=approval_policy,
            max_steps=max_steps_planner,
            max_new_tokens=max_new_tokens,
            read_only=True,
            role="planner",
        )
        self.coder = MiniAgent(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            approval_policy=approval_policy,
            max_steps=max_steps_coder,
            max_new_tokens=max_new_tokens,
            role="coder",
        )
        self.tester = MiniAgent(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            approval_policy=approval_policy,
            max_steps=max_steps_tester,
            max_new_tokens=max_new_tokens,
            read_only=False,
            role="tester",
        )
        self.reviewer = MiniAgent(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            approval_policy=approval_policy,
            max_steps=max_steps_reviewer,
            max_new_tokens=max_new_tokens,
            read_only=True,
            role="reviewer",
        )
        self.max_review_cycles = max_review_cycles
        self.graph = build_supervisor_graph(
            self.planner, self.coder, self.tester, self.reviewer,
            max_review_cycles=max_review_cycles,
        )
        self.workspace = workspace
        self.approval_policy = approval_policy
        self.session = {"id": "multi-agent"}
        self.session_path = "(multi-agent mode)"

    def memory_text(self):
        return "\n".join([
            "Memory:",
            "- planner notes:",
            self.planner.memory_text(),
            "- coder notes:",
            self.coder.memory_text(),
            "- tester notes:",
            self.tester.memory_text(),
            "- reviewer notes:",
            self.reviewer.memory_text(),
        ])

    def reset(self):
        self.planner.reset()
        self.coder.reset()
        self.tester.reset()
        self.reviewer.reset()

    def ask(self, user_message):
        from mini_coding_agent.observability import WorkflowEvent, event_bus

        run_id = uuid.uuid4().hex[:12]
        self.planner.run_id = run_id
        self.coder.run_id = run_id
        self.tester.run_id = run_id
        self.reviewer.run_id = run_id
        self.graph = build_supervisor_graph(
            self.planner, self.coder, self.tester, self.reviewer, run_id=run_id
        )

        event_bus.publish(WorkflowEvent.now(run_id, "system", "run_start", {"user_message": user_message}))

        initial_state = {
            "user_message": user_message,
            "plan": None,
            "test_report": None,
            "review_feedback": None,
            "review_verdict": None,
            "review_cycles": 0,
            "final_answer": None,
        }
        final_state = self.graph.invoke(initial_state)  # type: ignore

        verdict = final_state.get("review_verdict", "unknown")
        status = "approved" if verdict == "approved" else "max_cycles"
        event_bus.publish(
            WorkflowEvent.now(
                run_id,
                "system",
                "run_end",
                {
                    "status": status,
                    "user_message": user_message,
                    "final_answer": final_state.get("final_answer"),
                    "review_cycles": final_state.get("review_cycles", 0),
                },
            )
        )
        return final_state["final_answer"]


