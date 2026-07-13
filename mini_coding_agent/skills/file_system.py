"""File system skills: list_files, read_file, write_file, patch_file."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from mini_coding_agent.skills.base import Skill


class ListFilesParams(BaseModel):
    path: str = Field(default=".", description="Directory path relative to workspace root")


class ReadFileParams(BaseModel):
    path: str = Field(description="File path relative to workspace root")
    start: int = Field(default=1, ge=1, description="Start line number (1-indexed)")
    end: int = Field(default=200, ge=1, description="End line number")

    @field_validator("end")
    @classmethod
    def end_ge_start(cls, v: int, info) -> int:
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v


class WriteFileParams(BaseModel):
    path: str = Field(description="File path relative to workspace root")
    content: str = Field(description="File content to write")


class PatchFileParams(BaseModel):
    path: str = Field(description="File path relative to workspace root")
    old_text: str = Field(min_length=1, description="Exact text block to replace")
    new_text: str = Field(description="Replacement text")


class ListFilesSkill(Skill):
    name = "list_files"
    description = "List files in the workspace."
    risky = False
    param_model = ListFilesParams

    def run(self, agent, args: dict) -> str:
        from mini_coding_agent.context import IGNORED_PATH_NAMES, IGNORED_FILE_NAMES

        path = agent.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        entries = [
            item
            for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
            if item.name not in IGNORED_PATH_NAMES and item.name not in IGNORED_FILE_NAMES
        ]
        lines = []
        for entry in entries[:200]:
            kind = "[D]" if entry.is_dir() else "[F]"
            lines.append(f"{kind} {entry.relative_to(agent.root)}")
        return "\n".join(lines) or "(empty)"


class ReadFileSkill(Skill):
    name = "read_file"
    description = "Read a UTF-8 file by line range."
    risky = False
    param_model = ReadFileParams

    def run(self, agent, args: dict) -> str:
        from mini_coding_agent.context import IGNORED_FILE_NAMES

        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        if path.name in IGNORED_FILE_NAMES:
            return (
                f"# {path.relative_to(agent.root)}\n"
                "   1: (This file is part of the agent's own test suite and is irrelevant to the current task.)"
            )
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(
            f"{number:>4}: {line}"
            for number, line in enumerate(lines[start - 1:end], start=start)
        )
        return f"# {path.relative_to(agent.root)}\n{body}"


class WriteFileSkill(Skill):
    name = "write_file"
    description = 'Write a text file. MUST use XML: <tool name="write_file" path="file.py"><content>...</content></tool>'
    risky = True
    param_model = WriteFileParams

    def run(self, agent, args: dict) -> str:
        path = agent.path(args["path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(agent.root)} ({len(content)} chars)"


class PatchFileSkill(Skill):
    name = "patch_file"
    description = 'Replace one exact text block in a file. MUST use XML: <tool name="patch_file" path="file.py"><old_text>...</old_text><new_text>...</new_text></tool>'
    risky = True
    param_model = PatchFileParams

    def run(self, agent, args: dict) -> str:
        path = agent.path(args["path"])
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
        return f"patched {path.relative_to(agent.root)}"
