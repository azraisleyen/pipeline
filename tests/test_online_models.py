import numpy as np

from src.online.request_models import OnlineFrameRequest


def test_online_request_to_frame_context_wires_translation_and_session():
    req = OnlineFrameRequest(
        frame_id=7,
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        user="team",
        frame_name="frame_7.jpg",
        prediction_id="pred_7",
        session_id="session-a",
        upstream_translation={"translation_x": 1, "translation_y": 2, "translation_z": 3},
        translation_valid=True,
        reference_info={"reference_dir": "refs"},
    )
    ctx = req.to_frame_context()
    assert ctx.resolved_frame_name() == "frame_7.jpg"
    assert ctx.resolved_prediction_id() == "pred_7"
    assert ctx.health_status["translation_valid"] is True
    assert ctx.upstream_translation["translation_z"] == 3
    assert ctx.session_info()["session_id"] == "session-a"
    assert ctx.session_info()["reference_dir"] == "refs"
