"""Delegate skill: spawn a bounded read-only child agent."""

from pydantic import BaseModel, Field

from mini_coding_agent.skills.base import Skill


class DelegateParams(BaseModel):
    task: str = Field(min_length=1, description="Task description for the child agent")
    max_steps: int = Field(default=3, ge=1, description="Maximum steps for the child agent")


class DelegateSkill(Skill):
    name = "delegate"
    description = "Ask a bounded read-only child agent to investigate."
    risky = False
    param_model = DelegateParams

    def run(self, agent, args: dict) -> str:
        from mini_coding_agent.agent import MiniAgent
        from mini_coding_agent.context import clip

        if agent.depth >= agent.max_depth:
            raise ValueError("delegate depth exceeded")
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        child = MiniAgent(
            model_client=agent.model_client,
            workspace=agent.workspace,
            session_store=agent.session_store,
            approval_policy="never",
            max_steps=int(args.get("max_steps", 3)),
            max_new_tokens=agent.max_new_tokens,
            depth=agent.depth + 1,
            max_depth=agent.max_depth,
            read_only=True,
        )
        child.session["memory"]["task"] = task
        child.session["memory"]["notes"] = [clip(agent.history_text(), 300)]
        return "delegate_result:\n" + child.ask(task)
