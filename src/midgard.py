import os
from dataset import Dataset
import logging

class Midgard(Dataset):
    '''Helper functions for the MIDGARD dataset.'''

    def __init__(self, logger: logging.Logger, sequence: str) -> None:
        midgard_path = os.environ['MIDGARD_PATH']
        super().__init__(midgard_path, logger, sequence)

    def get_default_sequence(self) -> str:
        return 'countryside-natural/north-narrow'
