from midgard import Midgard

class RunConfig:
    def __init__(
        self,
        sequence: str,
        debug: bool,
        prepare_dataset: bool,
        evaluate: bool,
        headless: bool,
        use_nn_detection: bool,
        mode: str
    ):
        self.sequence = sequence
        self.debug = debug
        self.prepare_dataset = prepare_dataset
        self.evaluate = evaluate
        self.headless = headless
        self.use_nn_detection = use_nn_detection
        self.mode = self.get_mode(mode)

    def get_mode(self, mode_key: str) -> Midgard.Mode:
        options = [mode.name for mode in Midgard.Mode]
        if mode_key not in options:
            options_str = ', '.join(options)
            raise ValueError(f'Mode {mode_key} is not a valid mode type, has to be one of {options_str}')

        return Midgard.Mode[mode_key]
