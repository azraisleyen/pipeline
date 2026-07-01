import numpy as np
from src.common.frame_context import FrameContext
from src.task2_position.task2_module import Task2Module
def test_task2_never_empty():
 out=Task2Module({'task2':{}}).process(FrameContext(np.zeros((2,2,3),dtype=np.uint8),1)); assert out and set(out[0])=={'translation_x','translation_y','translation_z'}
