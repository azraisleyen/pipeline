import numpy as np
from src.common.frame_context import FrameContext
from src.common.packet_builder import PacketBuilder
def test_packet_builder_fields():
 p=PacketBuilder().build(FrameContext(np.zeros((4,4,3),dtype=np.uint8),1),[],[{'translation_x':0.0,'translation_y':0.0,'translation_z':0.0}],[])
 assert set(p)=={'id','user','frame','detected_objects','detected_translations','detected_undefined_objects'}
