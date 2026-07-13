"""Skills package: built-in and plugin-discoverable tools for Coding-Agent."""

from mini_coding_agent.skills.base import Skill, SkillRegistry
from mini_coding_agent.skills.file_system import (
    ListFilesSkill,
    ReadFileSkill,
    WriteFileSkill,
    PatchFileSkill,
)
from mini_coding_agent.skills.search import SearchSkill
from mini_coding_agent.skills.shell import RunShellSkill
from mini_coding_agent.skills.delegate import DelegateSkill

__all__ = [
    "Skill",
    "SkillRegistry",
    "ListFilesSkill",
    "ReadFileSkill",
    "WriteFileSkill",
    "PatchFileSkill",
    "SearchSkill",
    "RunShellSkill",
    "DelegateSkill",
]
