import logging
from typing import TypedDict

from mini_coding_agent.agent import MiniAgent

class SupervisorState(TypedDict):
    user_message: str
    plan: str | None
    test_report: str | None
    review_feedback: str | None
    review_verdict: str | None
    review_cycles: int
    final_answer: str | None


def _parse_reviewer_verdict(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    lower = raw.lower()
    if lower.startswith("approved"):
        return "approved", raw
    if lower.startswith("needs_fix"):
        feedback = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return "needs_fix", feedback
    return "needs_fix", raw


def build_supervisor_graph(planner: MiniAgent, coder: MiniAgent, tester: MiniAgent, reviewer: MiniAgent):
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(SupervisorState)

    def supervisor_node(state: SupervisorState):
        return {}

    def planner_node(state: SupervisorState):
        plan = planner.ask(state["user_message"])
        return {"plan": plan}

    def coder_node(state: SupervisorState):
        parts = [f"Plan:\n{state['plan']}"]
        if state.get("review_feedback"):
            parts.append(f"Review feedback:\n{state['review_feedback']}")
        parts.append(f"Implement: {state['user_message']}")
        prompt = "\n\n".join(parts)
        result = coder.ask(prompt)
        return {"final_answer": result}

    def tester_node(state: SupervisorState):
        prompt = (
            f"Original request: {state['user_message']}\n\n"
            f"Implementation:\n{state['final_answer']}\n\n"
            "Run tests, validate functionality, check edge cases, and return a test report."
        )
        report = tester.ask(prompt)
        return {"test_report": report}

    logger = logging.getLogger("mca.multi")

    def reviewer_node(state: SupervisorState):
        prompt = (
            f"Original request: {state['user_message']}\n\n"
            f"Implementation:\n{state['final_answer']}\n\n"
            f"Test report:\n{state['test_report']}\n\n"
            "Return <final>approved</final> or <final>needs_fix: [detailed feedback]</final>."
        )
        raw = reviewer.ask(prompt)
        verdict, feedback = _parse_reviewer_verdict(raw)
        logger.info("reviewer_verdict verdict=%s cycle=%d feedback_chars=%d", verdict, state["review_cycles"] + 1, len(feedback))
        return {
            "review_verdict": verdict,
            "review_feedback": feedback,
            "review_cycles": state["review_cycles"] + 1,
        }

    def supervisor_decision(state: SupervisorState):
        if state["review_verdict"] == "approved":
            logger.info("supervisor_decision approved cycle=%d", state["review_cycles"])
            return "approved"
        if state["review_cycles"] >= 3:
            logger.info("supervisor_decision max_cycles reached=%d", state["review_cycles"])
            return "max_cycles"
        logger.info("supervisor_decision needs_fix cycle=%d", state["review_cycles"])
        return "needs_fix"

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("planner", planner_node)
    builder.add_node("coder", coder_node)
    builder.add_node("tester", tester_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "planner")
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
    ):
        self.planner = MiniAgent(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            approval_policy=approval_policy,
            max_steps=max_steps_planner,
            read_only=True,
            role="planner",
        )
        self.coder = MiniAgent(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            approval_policy=approval_policy,
            max_steps=max_steps_coder,
            role="coder",
        )
        self.tester = MiniAgent(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            approval_policy=approval_policy,
            max_steps=max_steps_tester,
            read_only=True,
            role="tester",
        )
        self.reviewer = MiniAgent(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            approval_policy=approval_policy,
            max_steps=max_steps_reviewer,
            read_only=True,
            role="reviewer",
        )
        self.graph = build_supervisor_graph(self.planner, self.coder, self.tester, self.reviewer)
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
        initial_state = {
            "user_message": user_message,
            "plan": None,
            "test_report": None,
            "review_feedback": None,
            "review_verdict": None,
            "review_cycles": 0,
            "final_answer": None,
        }
        final_state = self.graph.invoke(initial_state) # type: ignore
        return final_state["final_answer"]


