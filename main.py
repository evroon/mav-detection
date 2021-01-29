import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from   lucas_kanade import LucasKanade
from   farneback import Farneback
from   midgard import Midgard

sequence = 'indoor-modern/sports-hall'
midgard = Midgard(sequence)

feature_pos = np.array([220.0, 280.0])
min_coords = np.zeros(2)
max_coords = midgard.capture_size[::-1] - np.ones(2)

try:
    while midgard.is_active():
        orig_frame = midgard.get_frame()

        flow_uv = midgard.get_flow_uv()
        flow_vis = midgard.get_flow_vis(flow_uv)
        flow_radial = midgard.get_flow_radial(flow_vis)
        ground_truth = midgard.get_midgard_annotation()

        feature_pos_int = feature_pos.astype(np.uint32).tolist()
        feature_pos += flow_uv[feature_pos_int[0], feature_pos_int[1], :]
        feature_pos_clipped = np.clip(feature_pos, min_coords, max_coords)

        if feature_pos[0] != feature_pos_clipped[0] or feature_pos[1] != feature_pos_clipped[1]:
            feature_pos = np.array(ground_truth.get_center())[::-1]
            feature_pos = np.clip(feature_pos, min_coords, max_coords)

        orig_frame = cv2.circle(orig_frame, tuple(feature_pos.astype(np.uint32)), 10, (0, 255, 0))
        orig_frame = cv2.rectangle(orig_frame, ground_truth.get_topleft(), ground_truth.get_bottomright(), (0, 0, 255), 3)

        top_frames = np.hstack((orig_frame, flow_vis))
        bottom_frames = np.hstack((flow_radial, np.zeros_like(orig_frame)))

        midgard.write(np.vstack((top_frames, bottom_frames)))

finally:
    midgard.release()
