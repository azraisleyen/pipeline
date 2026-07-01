from dataclasses import dataclass
@dataclass
class RuntimeState:
    initialized: bool=False
    processed_frames: int=0
