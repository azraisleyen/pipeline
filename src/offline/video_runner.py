from src.common.frame_context import FrameContext
from src.io.frame_sources import VideoFrameSource
class OfflineVideoRunner:
    def __init__(self, orchestrator): self.orchestrator=orchestrator
    def run(self, video_path, max_frames=None):
        packets=[]
        for i,name,fr in VideoFrameSource(video_path):
            if max_frames and len(packets)>=max_frames: break
            packets.append(self.orchestrator.process_frame(FrameContext(frame=fr,frame_id=i,frame_name=name)))
        return packets
