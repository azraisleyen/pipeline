from dataclasses import dataclass
@dataclass
class Task3State:
    initialized: bool=False
    frame_count: int=0
    def reset(self): self.initialized=False; self.frame_count=0
