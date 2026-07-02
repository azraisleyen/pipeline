import numpy as np

from src.common.frame_context import FrameContext
from src.task3_reference.task3_module import Task3Module


class FakeAdapter:
    def __init__(self):
        self.initialized_with = []
        self.reset_count = 0

    def initialize(self, session_info=None):
        self.initialized_with.append(dict(session_info or {}))

    def process(self, context):
        return []

    def reset(self):
        self.reset_count += 1


def test_task3_initializes_with_session_reference_dir(tmp_path):
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    module = Task3Module({"repo_root": str(tmp_path), "task3": {"reference_dir": str(ref_dir)}})
    fake = FakeAdapter()
    module.adapter = fake
    ctx = FrameContext(np.zeros((2, 2, 3), dtype=np.uint8), 1, session_id="s1", reference_info={"reference_dir": str(ref_dir)})
    assert module.process(ctx) == []
    assert fake.initialized_with[0]["session_id"] == "s1"
    assert fake.initialized_with[0]["reference_dir"] == str(ref_dir)
