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
        evaluate: bool,
        headless: bool,
        use_nn_detection: bool,
        mode: str
    ):
        self.logger = logger
        self.sequence = sequence
        self.debug = debug
        self.prepare_dataset = prepare_dataset
        self.evaluate = evaluate
        self.headless = headless
        self.use_nn_detection = use_nn_detection
        self.mode = self.get_mode(mode)
        self.results: dict = dict()

    def get_mode(self, mode_key: str) -> Midgard.Mode:
        options = [mode.name for mode in Midgard.Mode]
        if mode_key not in options:
            options_str = ', '.join(options)
            raise ValueError(
                f'Mode {mode_key} is not a valid mode type, has to be one of {options_str}'
            )

        return Midgard.Mode[mode_key]

    def __iter__(self) -> Iterator[Any]:
        return iter([
            self.sequence,
            self.debug,
            self.prepare_dataset,
            self.evaluate,
            self.headless,
            self.use_nn_detection,
            self.mode,
            self.results['ious'],
        ])

    def save_to_csv(self, filename: str) -> None:
        with open(filename, 'wb') as csv_file:
            wr = csv.writer(csv_file, delimiter=',')
            for cdr in self:
                print(cdr)
                wr.writerow(list(cdr))
