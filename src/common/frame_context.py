from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FrameContext:
    """Per-frame runtime context shared by all tasks.

    The core pipeline still processes exactly one image at a time, but online
    simulation needs enough metadata to preserve session state, reuse Task 3
    references, and pass upstream Task 2 translation signals through safely.
    """

    frame: np.ndarray
    frame_id: int
    frame_name: str | None = None
    prediction_id: str | None = None
    user: str = ""
    timestamp: float | None = None
    session_id: str | None = None
    frame_index: int | None = None
    image_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    upstream_translation: dict[str, float] | None = None
    health_status: dict[str, Any] = field(default_factory=dict)
    reference_info: dict[str, Any] = field(default_factory=dict)

    def resolved_frame_name(self) -> str:
        return self.frame_name or f"frame_{self.frame_id:06d}"

    def resolved_prediction_id(self) -> str:
        return self.prediction_id or f"prediction_{self.frame_id:06d}"

    def session_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        metadata_session = self.metadata.get("session_info")
        if isinstance(metadata_session, dict):
            info.update(metadata_session)
        if self.session_id:
            info.setdefault("session_id", self.session_id)
        if self.reference_info:
            info.setdefault("reference_info", self.reference_info)
            if self.reference_info.get("reference_dir"):
                info.setdefault("reference_dir", self.reference_info["reference_dir"])
        return info
