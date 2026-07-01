from dataclasses import dataclass, field
from typing import Any
import numpy as np
@dataclass
class FrameContext:
    frame: np.ndarray
    frame_id: int
    frame_name: str|None=None
    prediction_id: str|None=None
    user: str=''
    timestamp: float|None=None
    metadata: dict[str,Any]=field(default_factory=dict)
    upstream_translation: dict[str,float]|None=None
    health_status: dict[str,Any]=field(default_factory=dict)
    def resolved_frame_name(self)->str: return self.frame_name or f'frame_{self.frame_id:06d}'
    def resolved_prediction_id(self)->str: return self.prediction_id or f'prediction_{self.frame_id:06d}'
