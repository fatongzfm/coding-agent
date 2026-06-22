"""Event bus for multi-agent workflow observability."""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_event_logger = logging.getLogger("mca.events")


@dataclass
class WorkflowEvent:
    """An event emitted during workflow execution."""

    run_id: str
    timestamp: str
    node: str
    event_type: str
    payload: dict = field(default_factory=dict)

    @classmethod
    def now(cls, run_id: str, node: str, event_type: str, payload: dict | None = None):
        return cls(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            node=node,
            event_type=event_type,
            payload=payload or {},
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "node": self.node,
            "event_type": self.event_type,
            "payload": self.payload,
        }


class EventBus:
    """In-memory pub-sub event bus for workflow observability with JSONL persistence."""

    def __init__(self, log_dir: str | Path = ".logs"):
        self._subscribers: list[Callable[[WorkflowEvent], None]] = []
        self._history: dict[str, list[WorkflowEvent]] = {}
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _log_file(self, run_id: str) -> Path:
        return self._log_dir / f"{run_id}.jsonl"

    def subscribe(self, callback: Callable[[WorkflowEvent], None]):
        """Register a callback to receive all events."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[WorkflowEvent], None]):
        """Remove a callback."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def publish(self, event: WorkflowEvent):
        """Publish an event to all subscribers, archive it, and persist to disk."""
        self._history.setdefault(event.run_id, []).append(event)
        # Persist to JSONL so logs survive server restarts.
        try:
            with self._log_file(event.run_id).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass
        # Also emit a human-readable line to the text log file.
        try:
            payload_str = json.dumps(event.payload, ensure_ascii=False, separators=(",", ":"))
            _event_logger.info(
                "event=%s node=%s run_id=%s payload=%s",
                event.event_type,
                event.node,
                event.run_id,
                payload_str,
            )
        except Exception:
            pass
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:
                # Never let observability break the main workflow
                pass

    def get_history(self, run_id: str) -> list[WorkflowEvent]:
        """Return archived events for a given run (in-memory + disk)."""
        # Load from disk if not yet in memory (e.g. after server restart).
        if run_id not in self._history:
            self._load_from_disk(run_id)
        return list(self._history.get(run_id, []))

    def _load_from_disk(self, run_id: str):
        """Load historical events for a run_id from its JSONL file."""
        path = self._log_file(run_id)
        if not path.is_file():
            return
        events = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        events.append(
                            WorkflowEvent(
                                run_id=data["run_id"],
                                timestamp=data["timestamp"],
                                node=data["node"],
                                event_type=data["event_type"],
                                payload=data.get("payload", {}),
                            )
                        )
                    except Exception:
                        continue
        except Exception:
            pass
        self._history[run_id] = events

    def clear_history(self, run_id: str | None = None):
        """Clear archived events. If run_id is None, clear everything."""
        if run_id is None:
            self._history.clear()
            for f in self._log_dir.glob("*.jsonl"):
                try:
                    f.unlink()
                except Exception:
                    pass
        else:
            self._history.pop(run_id, None)
            try:
                self._log_file(run_id).unlink(missing_ok=True)
            except Exception:
                pass


# Global singleton – imported by server and multi_agent.
event_bus = EventBus()


def make_emitter(run_id: str | None):
    """Return a convenience emitter bound to a run_id (or a no-op if run_id is None)."""

    def emit(node: str, event_type: str, payload: dict | None = None):
        if run_id is None:
            return
        event_bus.publish(WorkflowEvent.now(run_id, node, event_type, payload))

    return emit
