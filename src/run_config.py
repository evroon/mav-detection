import logging
from typing import Iterator, Any
from enum import Enum

from midgard import Midgard


class RunConfig:
    class Mode(Enum):
        APPEARANCE_RGB = 0,
        FLOW_UV = 1,
        FLOW_RADIAL = 2,
        FLOW_PROCESSED_YOLO = 3,
        FLOW_PROCESSED_CLUSTERING = 4,

        def __str__(self) -> str:
            return super().__str__().replace('Mode.', '')

    def __init__(
        self,
        logger: logging.Logger,
        sequence: str,
        debug: bool,
        prepare_dataset: bool,
        validate: bool,
        headless: bool,
        data_to_yolo: bool,
        mode: str
    ):
        self.logger = logger
        self.sequence = sequence
        self.debug = debug
        self.prepare_dataset = prepare_dataset
        self.validate = validate
        self.headless = headless
        self.data_to_yolo = data_to_yolo
        self.mode = self.get_mode(mode)
        self.results: dict = dict()

    def uses_nn_for_detection(self) -> bool:
        return self.mode in [
            RunConfig.Mode.FLOW_UV,
            RunConfig.Mode.FLOW_RADIAL,
            RunConfig.Mode.FLOW_PROCESSED_YOLO
        ]

    def get_mode(self, mode_key: str) -> Mode:
        """Converts a str key to the Mode object.

        Args:
            mode_key (str): key that specifies the Mode to return

        Returns:
            Midgard.Mode: The resulting mode
        """
        options = [mode.name for mode in RunConfig.Mode]
        if mode_key not in options:
            options_str = ', '.join(options)
            raise ValueError(
                f'Mode {mode_key} is not a valid mode type, has to be one of {options_str}'
            )

        return RunConfig.Mode[mode_key]

    def __str__(self) -> str:
        return f'{self.sequence}/{self.mode}'

    def __iter__(self) -> Iterator[Any]:
        return iter([
            self.sequence,
            self.debug,
            self.prepare_dataset,
            self.validate,
            self.headless,
            self.data_to_yolo,
            self.mode,
            *self.results,
        ])
