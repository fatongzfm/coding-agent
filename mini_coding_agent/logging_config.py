"""Structured logging setup for Mini-Coding-Agent."""

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_dir: str | Path | None = None) -> None:
    """Configure stdlib logging for the agent.

    Args:
        level: Console logging level (default ``INFO``).
        log_dir: Optional directory for file-based logs.
    """
    handlers: list[logging.Handler] = []

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handlers.append(console)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "mini_coding_agent.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers,
        force=True,
    )
