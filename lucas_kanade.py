import numpy as np
import cv2
import argparse
import utils
import matplotlib.pyplot as plt
from typing import Optional

plt.rcParams['axes.axisbelow'] = True


class LucasKanade:
    def __init__(self, old_frame: np.ndarray) -> None:
        # Script based on: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

        self.old_frame = old_frame
        self.num_corners = 2000
        self.minimum_num_corners = self.num_corners // 3
        self.total_num_corners = self.num_corners + self.minimum_num_corners
        self.corners = np.zeros((self.total_num_corners, 2), dtype=np.int)
        self.num_features = 0
        self.features = []

        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=self.num_corners,
                                   qualityLevel=0.2,
                                   minDistance=7,
                                   blockSize=7)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(21, 21),
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # Create some random colors
        self.color = np.random.randint(0, 255, (self.total_num_corners, 3))

    def process(self, frame: np.ndarray):
        self.mask = np.zeros_like(self.old_frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.old_gray = cv2.cvtColor(self.old_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        if np.sum(self.old_gray) < 1:
            self.old_frame = frame
            return [], []

        if len(self.features) < self.minimum_num_corners:
            # Take first frame and find corners in it
            new_features = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            self.features = []

            for feature in new_features:
                self.features.append(feature[0])

        self.old_frame = frame

        # Calculate optical flow
        old_features = self.features
        good_new, status, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, np.array(old_features), None, **self.lk_params)
        self.features = good_new[status[:, 0] == 1, :].tolist()
        old_features = np.array(old_features)[status[:, 0] == 1, :].tolist()

        return np.array(old_features), np.array(self.features)
