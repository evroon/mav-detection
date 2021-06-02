from typing import Dict, Tuple, List, Optional, cast, Any, Iterator
import utils

class FrameResult:
    def __init__(self) -> None:
        self.time = 0.0
        self.tpr = 0.0
        self.fpr = 0.0
        self.sky_tpr = 0.0
        self.sky_fpr = 0.0
        self.drone_size_pixels = 0.0
        self.drone_flow_pixels = (0.0, 0.0)
        self.foe_dense = (0.0, 0.0)
        self.foe_gt = (0.0, 0.0)
