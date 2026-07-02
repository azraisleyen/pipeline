from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OnlineFrameResponse:
    packet: dict[str, Any]
    session_id: str | None = None
    frame_id: int | None = None
    submitted: bool = False
    submit_response: dict[str, Any] = field(default_factory=dict)
