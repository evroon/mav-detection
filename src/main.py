import argparse
import logging
from typing import List

from processor import Processor
from validator import Validator
from run_config import RunConfig

def execute(config: RunConfig) -> None:
    """Execute a configuration.

    Args:
        config (RunConfig): the configuration to run
    """
    config.logger.info(f'Starting: {config}')
    if config.validate and config.uses_nn_for_detection():
        validator = Validator(config)
        validator.run_validation()
    else:
        processor = Processor(config)
        try:
            if config.prepare_dataset:
                processor.convert(config.mode)
            elif config.data_to_yolo:
                processor.annotations_to_yolo()
            elif config.undistort:
                processor.undistort()
            else:
                detection_results = processor.run_detection()

            if config.validate and not config.uses_nn_for_detection():
                validator = Validator(config)
                validator.run_validation(detection_results)

        finally:
            processor.release()

def run_all(logger: logging.Logger, args: argparse.Namespace) -> None:
    settings = RunConfig.get_settings()
    dataset = 'MIDGARD'
    debug = True
    prepare_dataset = False
    validate = True
    headless = args.headless
    data_to_yolo = False
    undistort = False

    modes = [str(RunConfig.Mode.FLOW_FOE_CLUSTERING)]
    validation_sequences = settings['validation_sequences']
    configs: List[RunConfig] = []

    for sequence in validation_sequences:
        for mode in modes:
            config: RunConfig = RunConfig(logger, dataset, sequence, debug, prepare_dataset, validate, headless, data_to_yolo, undistort, mode)
            configs.append(config)
            execute(config)

def get_logger() -> logging.Logger:
    """Creates a logger object

    Returns:
        logging.Logger: the result logger
    """
    level = logging.INFO if args.debug else logging.DEBUG
    logging.basicConfig(filename='main.log',
                        filemode='a',
                        format='%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=level)

    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler())
    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects MAVs in the dataset using optical flow.')
    parser.add_argument('--dataset',            type=str, help='dataset to process', default='midgard')
    parser.add_argument('--sequence',           type=str, help='sequence to process', default='countryside-natural/north-narrow')
    parser.add_argument('--mode',               type=str, help='mode to use, see RunConfig.Mode', default='APPEARANCE_RGB')
    parser.add_argument('--debug',              action='store_true', help='whether to debug or not')
    parser.add_argument('--prepare-dataset',    action='store_true', help='prepares the YOLOv4 training dataset')
    parser.add_argument('--validate',           action='store_true', help='validate the detection results')
    parser.add_argument('--headless',           action='store_true', help='do not use UIs')
    parser.add_argument('--run-all',            action='store_true', help='run all configurations')
    parser.add_argument('--data-to-yolo',       action='store_true', help='convert annotations to the YOLO format')
    parser.add_argument('--undistort',          action='store_true', help='undistort original images')
    args = parser.parse_args()

    logger = get_logger()

    if args.run_all:
        run_all(logger, args)
    else:
        config: RunConfig = RunConfig(logger, args.dataset, args.sequence, args.debug, args.prepare_dataset, args.validate, args.headless, args.data_to_yolo, args.undistort, args.mode)
        execute(config)
