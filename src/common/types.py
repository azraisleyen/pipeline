from dataclasses import dataclass
from typing import Any
@dataclass(frozen=True)
class BBox: x1:int; y1:int; x2:int; y2:int
@dataclass
class TaskResult: items:list[dict[str,Any]]
