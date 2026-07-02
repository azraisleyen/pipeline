import numpy as np

from src.common.frame_context import FrameContext
from src.task2_position.task2_module import Task2Module


def test_task2_uses_valid_upstream_translation():
    module = Task2Module({"task2": {}})
    ctx = FrameContext(
        np.zeros((2, 2, 3), dtype=np.uint8),
        1,
        upstream_translation={"translation_x": 4, "translation_y": 5, "translation_z": 6},
        health_status={"translation_valid": True},
    )
    assert module.process(ctx) == [{"translation_x": 4.0, "translation_y": 5.0, "translation_z": 6.0}]


def test_task2_invalid_upstream_falls_back_to_last_valid_then_default():
    module = Task2Module({"task2": {"default_translation": {"translation_x": 0, "translation_y": 0, "translation_z": 0}}})
    valid = FrameContext(
        np.zeros((2, 2, 3), dtype=np.uint8),
        1,
        upstream_translation={"translation_x": 1, "translation_y": 2, "translation_z": 3},
        health_status={"translation_valid": True},
    )
    assert module.process(valid)[0]["translation_x"] == 1.0
    invalid = FrameContext(np.zeros((2, 2, 3), dtype=np.uint8), 2, upstream_translation=None, health_status={"translation_valid": False})
    assert module.process(invalid)[0]["translation_x"] == 1.0
    module.reset()
    assert module.process(invalid)[0] == {"translation_x": 0.0, "translation_y": 0.0, "translation_z": 0.0}
