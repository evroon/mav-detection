import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import utils

from midgard import Midgard
from midgard_converter import MidgardConverter
from evaluate import Validator
from run_config import RunConfig

def execute(config):
    if config.evaluate and config.use_nn_detection:
        validator = Validator()
        validator.run_validation(config.sequence)
    else:
        converter = MidgardConverter(config.sequence, config.debug, config.headless)
        try:
            if config.prepare_dataset:
                converter.convert(config.mode)
            else:
                converter.run_detection()
                detection_results = converter.get_results()

            if config.evaluate and config.use_nn_detection:
                validator = Validator()
                validator.run_validation(config.sequence)


        finally:
            converter.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects MAVs in the MIDGARD dataset using optical flow.')
    parser.add_argument('--sequence', type=str, help='sequence to process', default='countryside-natural/north-narrow')
    parser.add_argument('--mode', type=str, help='mode to use, see Midgard.Mode', default='APPEARANCE_RGB')
    parser.add_argument('--debug', action='store_true', help='whether to debug or not')
    parser.add_argument('--prepare-dataset', action='store_true', help='prepares the YOLOv4 training dataset')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the detection results')
    parser.add_argument('--headless', action='store_true', help='do not use UIs')
    parser.add_argument('--use-nn-detection', action='store_true', help='use neural network based approaches for detection')
    args = parser.parse_args()

    config: RunConfig = RunConfig(args.sequence, args.debug, args.prepare_dataset, args.evaluate, args.headless, args.use_nn_detection, args.mode)
    execute(config)
