"""Shell execution skill."""

import subprocess

from pydantic import BaseModel, Field

from mini_coding_agent.skills.base import Skill


class RunShellParams(BaseModel):
    command: str = Field(min_length=1, description="Shell command to run")
    timeout: int = Field(default=20, ge=1, le=120, description="Timeout in seconds")


class RunShellSkill(Skill):
    name = "run_shell"
    description = "Run a shell command in the repo root."
    risky = True
    param_model = RunShellParams

    def run(self, agent, args: dict) -> str:
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        result = subprocess.run(
            command,
            cwd=agent.root,
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
