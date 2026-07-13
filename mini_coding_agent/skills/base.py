"""Skill framework with Pydantic schema validation and plugin registration."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar

from pydantic import BaseModel, ValidationError


class SkillRegistry:
    """Global registry for skills."""

    _skills: ClassVar[dict[str, Skill]] = {}

    @classmethod
    def register(cls, skill: Skill) -> Skill:
        cls._skills[skill.name] = skill
        return skill

    @classmethod
    def get(cls, name: str) -> Skill | None:
        return cls._skills.get(name)

    @classmethod
    def list_skills(cls) -> dict[str, Skill]:
        return dict(cls._skills)

    @classmethod
    def clear(cls) -> None:
        cls._skills.clear()

    @classmethod
    def discover_builtin(cls) -> None:
        """Auto-discover and register all built-in skills from mini_coding_agent.skills package."""
        import mini_coding_agent.skills as skills_pkg

        pkg_path = Path(skills_pkg.__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(pkg_path)]):
            if module_name.startswith("_"):
                continue
            try:
                importlib.import_module(f"mini_coding_agent.skills.{module_name}")
            except Exception:
                # Best-effort: skip modules that fail to import
                pass

    @classmethod
    def discover_from_directory(cls, directory: str | Path) -> None:
        """Discover skills from an external directory."""
        import sys

        path = Path(directory).resolve()
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = py_file.stem
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception:
                pass


class Skill(ABC):
    """Abstract base class for a skill (tool).

    Each skill declares:
    - name: unique identifier
    - description: human-readable description for the LLM
    - risky: whether the skill requires approval
    - param_model: a Pydantic BaseModel defining the input schema
    """

    name: str = ""
    description: str = ""
    risky: bool = False
    param_model: type[BaseModel] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Auto-register concrete skill subclasses when they are defined
        if (
            cls.name
            and ABC not in cls.__bases__
            and not getattr(cls, "_abstract", False)
        ):
            SkillRegistry.register(cls())

    def validate_and_run(self, agent: Any, raw_args: dict[str, Any] | None) -> str:
        """Validate arguments using Pydantic, then execute the skill."""
        args = raw_args or {}
        if self.param_model is not None:
            try:
                validated = self.param_model.model_validate(args)
            except ValidationError as exc:
                return f"error: invalid arguments for {self.name}: {exc.errors()[0]['msg']}"
            # Convert model to dict for the run method
            run_args = validated.model_dump()
        else:
            run_args = args
        return self.run(agent, run_args)

    @abstractmethod
    def run(self, agent: Any, args: dict[str, Any]) -> str:
        """Execute the skill. Must be implemented by subclasses."""
        raise NotImplementedError

    def get_schema_dict(self) -> dict[str, Any]:
        """Return JSON-schema-like dict for LLM prompt generation."""
        if self.param_model is not None:
            return self.param_model.model_json_schema()
        return {}

    def get_schema_lines(self) -> list[str]:
        """Return human-readable schema lines for the prompt."""
        lines = []
        if self.param_model is not None:
            for field_name, field_info in self.param_model.model_fields.items():
                annotation = field_info.annotation
                default = field_info.default
                if default is not inspect.Parameter.empty and default is not None:
                    lines.append(f"  {field_name}: {annotation} = {default}")
                else:
                    lines.append(f"  {field_name}: {annotation}")
        return lines
