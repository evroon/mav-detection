import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from lucas_kanade import LucasKanade
from farneback import Farneback
from midgard import Midgard
from detector import Detector
import argparse

parser = argparse.ArgumentParser(description='Detects MAVs in the MIDGARD dataset using optical flow.')
parser.add_argument('--sequence', type=str, help='sequence to process', default='countryside-natural/north-narrow')
parser.add_argument('--debug', type=bool, help='debug', default=False)
args = parser.parse_args()

midgard = Midgard(args.sequence, args.debug)
detector = Detector(midgard)

try:
    while midgard.is_active():
        orig_frame = midgard.get_frame()

        detector.get_affine_matrix()
        flow_uv_warped_vis, flow_uv_warped_mag_vis = detector.flow_vec_subtract()

        if args.debug:
            flow_diff_vis, blocks_vis = detector.block_method()
            cluster_vis, segment = detector.clustering(detector.flow_uv_warped_mag)
            blocks_vis, _ = detector.clustering(detector.flow_diff_mag)
            summed_mag = detector.get_history()
            detector.draw()

            top_frames = np.hstack((orig_frame, flow_diff_vis, flow_uv_warped_vis, summed_mag))
            bottom_frames = np.hstack((midgard.flow_vis, blocks_vis, cluster_vis, summed_mag))
            midgard.write(np.vstack((top_frames, bottom_frames)))
        else:
            cluster_vis = detector.clustering(detector.flow_uv_warped_mag, True)
            midgard.write(cluster_vis)

finally:
    midgard.release()
