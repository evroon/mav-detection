import logging
import json
from typing import Iterator, Any, Dict, List, cast
from enum import Enum

from dataset import Dataset
from midgard import Midgard
from sim_data import SimData


class RunConfig:
    class Mode(Enum):
        APPEARANCE_RGB = 0,
        FLOW_UV = 1,
        FLOW_RADIAL = 2,
        FLOW_FOE_YOLO = 3,
        FLOW_FOE_CLUSTERING = 4,

        def __str__(self) -> str:
            return super().__str__().replace('Mode.', '')

    class DatasetType(Enum):
        MIDGARD = 0,
        SIMULATION = 1,
        EXPERIMENT = 2

        def __str__(self) -> str:
            return super().__str__().replace('DatasetType.', '')

    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        with open('settings.json', 'r') as f:
            return cast(Dict[str, Any], json.load(f))

    def __init__(
        self,
        logger: logging.Logger,
        dataset: str,
        sequence: str,
        debug: bool,
        prepare_dataset: bool,
        validate: bool,
        headless: bool,
        data_to_yolo: bool,
        undistort: bool,
        mode: str
    ):
        self.logger = logger
        self.dataset = dataset
        self.sequence = sequence
        self.debug = debug
        self.prepare_dataset = prepare_dataset
        self.validate = validate
        self.headless = headless
        self.data_to_yolo = data_to_yolo
        self.undistort = undistort
        self.mode = self.get_mode(mode)
        self.results: dict = dict()
        self.settings = RunConfig.get_settings()

    def get_all_sequences(self) -> List[str]:
        sequences = self.settings['train_sequences']
        for seq in self.settings['validation_sequences']:
            sequences.append(seq)
        return cast(List[str], sequences)

    def uses_nn_for_detection(self) -> bool:
        return self.mode in [
            RunConfig.Mode.FLOW_UV,
            RunConfig.Mode.FLOW_RADIAL,
            RunConfig.Mode.FLOW_FOE_YOLO
        ]

    def get_mode(self, mode_key: str) -> Mode:
        """Converts a str key to the Mode object.

        Args:
            mode_key (str): key that specifies the Mode to return

        Returns:
            RunConfig.Mode: The resulting mode
        """
        options = [mode.name for mode in RunConfig.Mode]
        if mode_key not in options:
            options_str = ', '.join(options)
            raise ValueError(
                f'Mode {mode_key} is not a valid mode type, has to be one of {options_str}'
            )

        return RunConfig.Mode[mode_key]

    def get_dataset_type(self, dataset_key: str) -> DatasetType:
        """Converts a str key to the DatasetType object.

        Args:
            dataset_key (str): key that specifies the DatasetType to return

        Returns:
            RunConfig.DatasetType: The resulting dataset type
        """
        options = [mode.name for mode in RunConfig.DatasetType]
        dataset_key = dataset_key.upper()
        if dataset_key not in options:
            options_str = ', '.join(options)
            raise ValueError(
                f'Dataset {dataset_key} is not a valid dataset type, has to be one of {options_str}'
            )

        return RunConfig.DatasetType[dataset_key]

    def get_dataset(self) -> Dataset:
        data_type = self.get_dataset_type(self.dataset)
        if data_type == RunConfig.DatasetType.MIDGARD:
            dataset: Dataset = Midgard(self.logger, self.sequence)
        elif data_type == RunConfig.DatasetType.SIMULATION:
            dataset = SimData(self.logger, self.sequence)
        else:
            dataset = Dataset('', self.logger, self.sequence)

        self.sequence = dataset.sequence
        return dataset

    def __str__(self) -> str:
        return f'{self.dataset}/{self.sequence}/{self.mode}'

    def __iter__(self) -> Iterator[Any]:
        return iter([
            self.dataset,
            self.sequence,
            self.debug,
            self.prepare_dataset,
            self.validate,
            self.headless,
            self.data_to_yolo,
            self.undistort,
            self.mode,
            *self.results,
        ])
