"""Search skill: search the workspace with rg or a simple fallback."""

import shutil
import subprocess

from pydantic import BaseModel, Field

from mini_coding_agent.skills.base import Skill


class SearchParams(BaseModel):
    pattern: str = Field(min_length=1, description="Search pattern")
    path: str = Field(default=".", description="Directory or file path to search in")


class SearchSkill(Skill):
    name = "search"
    description = "Search the workspace with rg or a simple fallback."
    risky = False
    param_model = SearchParams

    def run(self, agent, args: dict) -> str:
        from mini_coding_agent.context import IGNORED_PATH_NAMES, IGNORED_FILE_NAMES

        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        path = agent.path(args.get("path", "."))

        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
                cwd=agent.root,
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
            and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(agent.root).parts)
            and item.name not in IGNORED_FILE_NAMES
        ]
        for file_path in files:
            for number, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if pattern.lower() in line.lower():
                    matches.append(f"{file_path.relative_to(agent.root)}:{number}:{line}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"
