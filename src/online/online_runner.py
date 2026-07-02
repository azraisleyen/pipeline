from __future__ import annotations

from typing import Any

from src.common.schema_validation import SchemaValidation
from src.online.response_models import OnlineFrameResponse
from src.pipeline.health_monitor import HealthMonitor


class OnlineRunner:
    def __init__(self, orchestrator, api_client, *, health_monitor: HealthMonitor | None = None, fail_fast: bool = False):
        self.orchestrator = orchestrator
        self.api_client = api_client
        self.health = health_monitor or getattr(orchestrator, "health", HealthMonitor())
        self.fail_fast = fail_fast
        self._active_session_id: str | None = None

    def process_request(self, request):
        context = request.to_frame_context() if hasattr(request, "to_frame_context") else request
        return self.orchestrator.process_frame(context)

    def run(self, *, max_frames: int | None = None) -> list[OnlineFrameResponse]:
        responses: list[OnlineFrameResponse] = []
        processed = 0
        while max_frames is None or processed < max_frames:
            try:
                request = self.api_client.fetch_frame()
            except Exception as exc:
                self.health.record_error(exc, component="online.fetch", recoverable=not self.fail_fast)
                if self.fail_fast:
                    raise
                break

            if request is None or getattr(request, "end_of_stream", False):
                break

            if request.session_id and self._active_session_id and request.session_id != self._active_session_id:
                self.orchestrator.finalize_session()
            if request.session_id:
                self._active_session_id = request.session_id

            try:
                packet = self.process_request(request)
                SchemaValidation.validate(packet)
                submit_response = self.api_client.submit_packet(packet)
                responses.append(
                    OnlineFrameResponse(
                        packet=packet,
                        session_id=request.session_id,
                        frame_id=request.frame_id,
                        submitted=True,
                        submit_response=submit_response,
                    )
                )
                processed += 1
            except Exception as exc:
                self.health.record_error(exc, component="online.process_submit", recoverable=not self.fail_fast, frame_id=getattr(request, "frame_id", None))
                if self.fail_fast:
                    raise
                break
        return responses
