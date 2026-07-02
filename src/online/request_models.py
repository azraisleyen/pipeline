from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from src.common.frame_context import FrameContext
from src.common.exceptions import PipelineError


@dataclass
class OnlineFrameRequest:
    frame_id: int
    frame: np.ndarray
    user: str = ""
    frame_name: str | None = None
    prediction_id: str | None = None
    session_id: str | None = None
    frame_index: int | None = None
    timestamp: float | None = None
    image_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    upstream_translation: dict[str, float] | None = None
    translation_valid: bool | None = None
    health_status: dict[str, Any] = field(default_factory=dict)
    reference_info: dict[str, Any] = field(default_factory=dict)
    end_of_stream: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, timeout_seconds: float = 10.0, token: str = "") -> "OnlineFrameRequest":
        frame = _decode_frame(payload, timeout_seconds=timeout_seconds, token=token)
        frame_id = int(payload.get("frame_id") or payload.get("frame_index") or payload.get("id") or 0)
        if frame_id <= 0:
            frame_id = int(payload.get("sequence", 1))
        upstream = payload.get("upstream_translation") or payload.get("translation") or payload.get("detected_translation")
        health = dict(payload.get("health_status") or {})
        if "translation_valid" in payload:
            health["translation_valid"] = bool(payload["translation_valid"])
        reference_info = dict(payload.get("reference_info") or {})
        if payload.get("reference_dir"):
            reference_info.setdefault("reference_dir", payload["reference_dir"])
        metadata = dict(payload.get("metadata") or {})
        if payload.get("session_info"):
            metadata["session_info"] = payload["session_info"]
        return cls(
            frame_id=frame_id,
            frame=frame,
            user=str(payload.get("user", "")),
            frame_name=payload.get("frame") or payload.get("frame_name"),
            prediction_id=payload.get("prediction_id") or payload.get("packet_id"),
            session_id=payload.get("session_id"),
            frame_index=payload.get("frame_index"),
            timestamp=payload.get("timestamp"),
            image_metadata=dict(payload.get("image_metadata") or {}),
            metadata=metadata,
            upstream_translation=upstream,
            translation_valid=health.get("translation_valid"),
            health_status=health,
            reference_info=reference_info,
            end_of_stream=bool(payload.get("end_of_stream", False)),
        )

    def to_frame_context(self) -> FrameContext:
        health = dict(self.health_status or {})
        if self.translation_valid is not None:
            health["translation_valid"] = bool(self.translation_valid)
        return FrameContext(
            frame=self.frame,
            frame_id=self.frame_id,
            frame_name=self.frame_name,
            prediction_id=self.prediction_id,
            user=self.user,
            timestamp=self.timestamp,
            session_id=self.session_id,
            frame_index=self.frame_index,
            image_metadata=dict(self.image_metadata),
            metadata=dict(self.metadata),
            upstream_translation=self.upstream_translation,
            health_status=health,
            reference_info=dict(self.reference_info),
        )


def _decode_frame(payload: dict[str, Any], *, timeout_seconds: float, token: str) -> np.ndarray:
    if isinstance(payload.get("frame"), np.ndarray):
        return payload["frame"]
    if payload.get("image_b64"):
        raw = base64.b64decode(payload["image_b64"])
        return _decode_image_bytes(raw)
    if payload.get("image_bytes"):
        return _decode_image_bytes(payload["image_bytes"])
    if payload.get("image_url"):
        return _download_image(payload["image_url"], timeout_seconds=timeout_seconds, token=token)
    if payload.get("frame_path"):
        import cv2
        frame = cv2.imread(str(payload["frame_path"]), cv2.IMREAD_COLOR)
        if frame is None:
            raise PipelineError(f"Unable to read frame_path: {payload['frame_path']}")
        return frame
    raise PipelineError("Online frame payload does not contain frame, image_b64, image_bytes, image_url, or frame_path")


def _decode_image_bytes(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    import cv2
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise PipelineError("Unable to decode image bytes from online payload")
    return frame


def _download_image(url: str, *, timeout_seconds: float, token: str) -> np.ndarray:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read()
    return _decode_image_bytes(raw)
