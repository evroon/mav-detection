import os
from datasets.dataset import Dataset
import logging

class VisDrone(Dataset):
    '''Helper functions for the VisDrone dataset.'''

    def __init__(self, logger: logging.Logger, sequence: str) -> None:
        vis_drone_path = os.environ['VIS_DRONE_PATH']
        super().__init__(vis_drone_path, logger, sequence, '', '/sequences')

    def get_default_sequence(self) -> str:
        return 'uav0000244_01440_v'
