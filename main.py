import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from lucas_kanade import LucasKanade
from farneback import Farneback
from midgard import Midgard
from midgard_converter import MidgardConverter
from evaluate import Validator
import argparse

parser = argparse.ArgumentParser(description='Detects MAVs in the MIDGARD dataset using optical flow.')
parser.add_argument('--sequence', type=str, help='sequence to process', default='countryside-natural/north-narrow')
parser.add_argument("--debug", action="store_true")
parser.add_argument("--prepare-dataset", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--mode", type=str, help='sequence to process', default='APPEARANCE_RGB')
args = parser.parse_args()

if args.evaluate:
    validator = Validator()
    validator.run_validation('countryside-natural/north-narrow')
else:
    try:
        converter = MidgardConverter(args.sequence, args.debug)
        if args.prepare_dataset:
            options = [color.name for color in Midgard.Mode]
            if args.mode not in options:
                options_str = ', '.join(options)
                raise ValueError(f'Mode {args.mode} is not a valid mode, has to be one of {options_str}')
            converter.process(Midgard.Mode[args.mode])
        else:
            converter = MidgardConverter(args.sequence, args.debug)
            converter.run()

    finally:
        converter.release()
