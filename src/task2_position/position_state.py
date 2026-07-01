from dataclasses import dataclass
@dataclass
class PositionState:
    last_valid: dict[str,float]|None=None
    def update(self, tr): self.last_valid=tr; return tr
    def reset(self): self.last_valid=None
