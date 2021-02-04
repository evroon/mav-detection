import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from lucas_kanade import LucasKanade
from farneback import Farneback
from midgard import Midgard
from detector import Detector

sequence = 'indoor-modern/sports-hall'
midgard = Midgard(sequence)
detector = Detector(midgard)

try:
    while midgard.is_active():
        orig_frame = midgard.get_frame()

        # while midgard.i < 100:
        #     orig_frame = midgard.get_frame()
        #     midgard.i += 1

        detector.get_affine_matrix()
        flow_uv_warped_vis, flow_uv_warped_mag_vis = detector.flow_vec_subtract()
        flow_diff_vis, blocks_vis = detector.block_method()
        cluster_vis = detector.clustering(detector.flow_uv_warped_mag)
        blocks_vis = detector.clustering(detector.flow_diff_mag)
        detector.draw()

        top_frames = np.hstack((orig_frame, flow_diff_vis, flow_uv_warped_vis))
        bottom_frames = np.hstack((midgard.flow_vis, blocks_vis, cluster_vis))
        midgard.write(np.vstack((top_frames, bottom_frames)))

finally:
    midgard.release()
