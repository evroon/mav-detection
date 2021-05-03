from typing import Dict, Tuple, List, Optional, cast, Any, Iterator
import utils

class FrameResult:
    def __init__(self) -> None:
        self.boxes: List[Tuple[str, float, utils.Rectangle]] = []
        self.data: Dict[str, Any] = {}

    def __iter__(self) -> Iterator[Any]:
        return iter(self.boxes)

    def add_box(self, name: str, confidence: float, rect: utils.Rectangle) -> None:
        self.boxes.append((name, confidence, rect))
