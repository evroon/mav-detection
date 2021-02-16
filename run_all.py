from run_config import RunConfig
from midgard import Midgard
from typing import List
from main import execute


debug = False
prepare_dataset = False
evaluate = True
headless = True

modes = [mode.name for mode in Midgard.Mode]
use_nn_detections = [True, False]
validation_sequences = [
    'indoor-modern/warehouse-interior',
    'semi-urban/island-north',
]
configs: List[RunConfig] = []

for sequence in validation_sequences:
    for mode in modes:
        for use_nn_detection in use_nn_detections:
            config: RunConfig = RunConfig(sequence, debug, prepare_dataset, evaluate, headless, use_nn_detection, mode)
            configs.append(config)
            execute(config)
