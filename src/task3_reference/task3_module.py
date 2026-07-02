from __future__ import annotations

from .session_reference_manager import SessionReferenceManager
from .task3_state import Task3State
from .v501_adapter import Task3V501Adapter


class Task3Module:
    def __init__(self, config):
        self.config = config
        self.state = Task3State()
        t = config.get("task3", {})
        self.reference_manager = SessionReferenceManager(
            t.get("reference_dir"),
            repo_root=config.get("repo_root", "."),
            require_exists=not t.get("allow_unavailable", False),
        )
        self.adapter = Task3V501Adapter(config)
        self._session_id = None

    def initialize(self, session_info=None):
        resolved = self.reference_manager.resolve(session_info)
        adapter_session = dict(session_info or {})
        if resolved.reference_dir is not None:
            adapter_session["reference_dir"] = str(resolved.reference_dir)
        if resolved.session_id:
            adapter_session["session_id"] = resolved.session_id
        self.adapter.initialize(adapter_session)
        self.state.initialized = True
        self._session_id = resolved.session_id

    def process(self, context):
        session_info = context.session_info() if hasattr(context, "session_info") else {}
        session_id = session_info.get("session_id")
        if self.state.initialized and session_id and self._session_id and session_id != self._session_id:
            self.reset()
        if not self.state.initialized and not self.config.get("task3", {}).get("allow_unavailable", False):
            self.initialize(session_info)
        self.state.frame_count += 1
        return self.adapter.process(context)

    def reset(self):
        self.adapter.reset()
        self.state.reset()
        self._session_id = None
