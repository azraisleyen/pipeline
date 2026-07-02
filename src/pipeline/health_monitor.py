from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HealthEvent:
    level: str
    component: str
    message: str
    recoverable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    def __init__(self) -> None:
        self.events: list[HealthEvent] = []

    def record_error(self, exc: Exception | str, component: str = "pipeline", recoverable: bool = False, **metadata: Any) -> None:
        self.events.append(
            HealthEvent(
                level="error",
                component=component,
                message=str(exc),
                recoverable=recoverable,
                metadata=dict(metadata),
            )
        )

    def record_warning(self, message: str, component: str = "pipeline", **metadata: Any) -> None:
        self.events.append(HealthEvent(level="warning", component=component, message=message, recoverable=True, metadata=dict(metadata)))

    def status(self) -> dict[str, Any]:
        blocking = [e for e in self.events if e.level == "error" and not e.recoverable]
        return {
            "ok": not blocking,
            "events": [e.__dict__ for e in self.events],
            "blocking_errors": [e.__dict__ for e in blocking],
        }

    def clear(self) -> None:
        self.events.clear()
