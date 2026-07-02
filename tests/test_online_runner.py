import numpy as np

from src.online.online_runner import OnlineRunner
from src.online.request_models import OnlineFrameRequest
from src.pipeline.orchestrator import PipelineOrchestrator


class FakeClient:
    def __init__(self):
        self.submitted = []
        self.requests = [
            OnlineFrameRequest(
                frame_id=1,
                frame=np.zeros((4, 4, 3), dtype=np.uint8),
                upstream_translation={"translation_x": 9, "translation_y": 8, "translation_z": 7},
                translation_valid=True,
                session_id="s",
            ),
            None,
        ]

    def fetch_frame(self):
        return self.requests.pop(0)

    def submit_packet(self, packet):
        self.submitted.append(packet)
        return {"ok": True}


def test_online_runner_fetch_process_submit_lifecycle():
    cfg = {"task1": {"allow_missing_models": True}, "task2": {}, "task3": {"allow_unavailable": True}, "pipeline": {"validate_schema": True}}
    client = FakeClient()
    responses = OnlineRunner(PipelineOrchestrator(cfg), client).run()
    assert len(responses) == 1
    assert client.submitted[0]["detected_translations"][0]["translation_x"] == 9.0
    assert responses[0].submitted is True
