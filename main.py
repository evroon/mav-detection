import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import utils
import logging
from typing import List

from midgard import Midgard
from midgard_converter import MidgardConverter
from evaluate import Validator
from run_config import RunConfig

def execute(config: RunConfig) -> None:
    if config.evaluate and config.use_nn_detection:
        validator = Validator(config)
        validator.run_validation()
    else:
        converter = MidgardConverter(config)
        try:
            if config.prepare_dataset:
                converter.convert(config.mode)
            else:
                converter.run_detection()
                detection_results = converter.get_results()

            if config.evaluate and config.use_nn_detection:
                validator = Validator(config)
                validator.run_validation(detection_results)

        finally:
            converter.release()

def run_all(logger: logging.Logger) -> None:
    debug = False
    prepare_dataset = False
    evaluate = True
    headless = True

    modes = [mode.name for mode in Midgard.Mode]
    use_nn_detections = [False]
    validation_sequences = [
        # 'indoor-modern/warehouse-interior',
        'semi-urban/island-north',
    ]
    configs: List[RunConfig] = []

    for sequence in validation_sequences:
        for mode in modes:
            for use_nn_detection in use_nn_detections:
                config: RunConfig = RunConfig(logger, sequence, debug, prepare_dataset, evaluate, headless, use_nn_detection, mode)
                configs.append(config)
                execute(config)

def get_logger() -> logging.Logger:
    level = logging.INFO if args.debug else logging.DEBUG
    logging.basicConfig(filename='main.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=level)

    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler())
    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects MAVs in the MIDGARD dataset using optical flow.')
    parser.add_argument('--sequence', type=str, help='sequence to process', default='countryside-natural/north-narrow')
    parser.add_argument('--mode', type=str, help='mode to use, see Midgard.Mode', default='APPEARANCE_RGB')
    parser.add_argument('--debug', action='store_true', help='whether to debug or not')
    parser.add_argument('--prepare-dataset', action='store_true', help='prepares the YOLOv4 training dataset')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the detection results')
    parser.add_argument('--headless', action='store_true', help='do not use UIs')
    parser.add_argument('--use-nn-detection', action='store_true', help='use neural network based approaches for detection')
    parser.add_argument('--run-all', action='store_true', help='run all configurations')
    args = parser.parse_args()

    logger = get_logger()

    if args.run_all:
        run_all(logger)
    else:
        config: RunConfig = RunConfig(logger, args.sequence, args.debug, args.prepare_dataset, args.evaluate, args.headless, args.use_nn_detection, args.mode)
        execute(config)
