from dataclasses import dataclass
from src.common.bbox_ops import center
@dataclass
class MotionTrack: center:tuple[float,float]; frame_id:int
class VehicleMotionEstimator:
    def __init__(self, threshold_px=12.0, max_lost_frames=5): self.threshold=float(threshold_px); self.max_lost=int(max_lost_frames); self.tracks={}
    def estimate(self, bbox, frame_id):
        c=center(bbox); prev=self.tracks.get('vehicle')
        self.tracks['vehicle']=MotionTrack(c,frame_id)
        if prev is None or frame_id-prev.frame_id>self.max_lost: return '0'
        dist=((c[0]-prev.center[0])**2+(c[1]-prev.center[1])**2)**0.5
        return '1' if dist>=self.threshold else '0'
    def reset(self): self.tracks.clear()
