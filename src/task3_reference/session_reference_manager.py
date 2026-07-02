from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.exceptions import ConfigError


@dataclass(frozen=True)
class ResolvedReferences:
    session_id: str | None
    reference_dir: Path | None
    reference_info: dict[str, Any]


class SessionReferenceManager:
    def __init__(self, reference_dir=None, repo_root: str | Path = ".", require_exists: bool = True):
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.repo_root = Path(repo_root)
        self.require_exists = require_exists

    def resolve(self, session_info=None) -> ResolvedReferences:
        session_info = dict(session_info or {})
        reference_info = dict(session_info.get("reference_info") or {})
        raw_dir = session_info.get("reference_dir") or reference_info.get("reference_dir") or self.reference_dir
        ref_dir = Path(raw_dir) if raw_dir else None
        if ref_dir is not None and not ref_dir.is_absolute():
            ref_dir = self.repo_root / ref_dir
        if self.require_exists and ref_dir is not None and not ref_dir.exists():
            raise ConfigError(f"Task3 reference directory is missing for session {session_info.get('session_id') or '<default>'}: {ref_dir}")
        return ResolvedReferences(session_id=session_info.get("session_id"), reference_dir=ref_dir, reference_info=reference_info)
