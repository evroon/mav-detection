import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from   lucas_kanade import LucasKanade
from   farneback import Farneback
from   midgard import Midgard

sequence = 'indoor-modern/sports-hall'
midgard = Midgard(sequence)

try:
    while midgard.is_active():
        orig_frame = midgard.get_frame()

        flow_uv = midgard.get_flow_uv()
        flow_vis = midgard.get_flow_vis(flow_uv)
        flow_radial = midgard.get_flow_radial(flow_vis)

        top_frames = np.hstack((orig_frame, flow_vis))
        bottom_frames = np.hstack((flow_radial, np.zeros_like(orig_frame)))

        midgard.write(np.vstack((top_frames, bottom_frames)))
        midgard.iterate()

finally:
    midgard.release()
