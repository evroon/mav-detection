import csv
import logging
from typing import Dict, Iterator, List, Any

from midgard import Midgard


class RunConfig:
    def __init__(
        self,
        logger: logging.Logger,
        sequence: str,
        debug: bool,
        prepare_dataset: bool,
        validate: bool,
        headless: bool,
        use_nn_detection: bool,
        mode: str
    ):
        self.logger = logger
        self.sequence = sequence
        self.debug = debug
        self.prepare_dataset = prepare_dataset
        self.validate = validate
        self.headless = headless
        self.use_nn_detection = use_nn_detection
        self.mode = self.get_mode(mode)
        self.results: dict = dict()

    def get_mode(self, mode_key: str) -> Midgard.Mode:
        options = [mode.name for mode in Midgard.Mode]
        mode_key = mode_key.replace('Mode.', '')
        if mode_key not in options:
            options_str = ', '.join(options)
            raise ValueError(
                f'Mode {mode_key} is not a valid mode type, has to be one of {options_str}'
            )

        return Midgard.Mode[mode_key]

    def __str__(self) -> str:
        return f'sequence: {self.sequence}, mode: {self.mode}'

    def __iter__(self) -> Iterator[Any]:
        return iter([
            self.sequence,
            self.debug,
            self.prepare_dataset,
            self.validate,
            self.headless,
            self.use_nn_detection,
            self.mode,
            *self.results,
        ])
