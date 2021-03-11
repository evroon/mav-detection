import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

from lucas_kanade import LucasKanade

plt.rcParams['axes.axisbelow'] = True


class FocusOfExpansion:
    def __init__(self, lucas_kanade: LucasKanade) -> None:
        # Script based on: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

        self.lucas_kanade = lucas_kanade

        # Create some random colors
        self.color = np.random.randint(0, 255, (self.lucas_kanade.total_num_corners, 3))

        self.time = 0
        self.trace = np.zeros((self.lucas_kanade.total_num_corners, 2000), dtype=np.int)
        self.roll_back = 20
        self.num_features = 0
        self.random_lines = np.random.randint(0, self.lucas_kanade.total_num_corners, self.lucas_kanade.total_num_corners)
        self.threshold = np.cos(15 * np.pi / 180.0)
        self.flow_per_distance = np.zeros((self.lucas_kanade.total_num_corners, 4)) # Store distance, flow magnitude pairs
        self.enable_plots = False

    def get_FOE(self, old_frame: np.ndarray, new_frame: np.ndarray) -> Tuple[float, float]:
        if np.sum(old_frame) < 1:
            return (np.nan, np.nan)

        old_features, new_features, status = self.lucas_kanade.get_features(new_frame)
        self.mask = np.zeros_like(old_frame)
        self.lines = []
        intersections = np.zeros((len(new_features), 2))

        # Draw the new tracks
        old: np.ndarray
        for i, (new, old) in enumerate(zip(new_features, utils.assert_type(old_features))):
            if status[i] != 1:
                continue

            a, b, c, d = [int(x) for x in [*new.ravel(), *old.ravel()]]
            color = self.color[i].tolist()

            # new_frame = cv2.circle(new_frame, (a, b), 4, color, -1)

            l = self.trace[i, 0] + 1
            self.trace[i, l] = c
            self.trace[i, l+1] = d
            self.trace[i, 0] += 2
            color = self.color[i].tolist()
            # m, n = self.trace[i, 1:3]
            self.num_features += 1

            # for j in range(3, l+2, 2):
                # self.mask = cv2.line(self.mask, (m, n), (self.trace[i, j], self.trace[i, j+1]), color, 2)
                # m, n = self.trace[i, j], self.trace[i, j+1]

            if l >= 3:
                k = 1 if l < 1 + self.roll_back * 2 else l - self.roll_back * 2
                a, b = self.trace[i, l:l+2]
                c, d = self.trace[i, k:k+2]

                diff_x = float(c) - float(a)
                diff_y = float(d) - float(b)

                diff = np.array([diff_x, diff_y])
                # length = np.linalg.norm(diff)

                # if length < 5: # or b < capture_size[1] // 4:
                    # continue

                new_xy = np.array([a, b]) + diff
                new_xy = new_xy.astype(np.uint16)

                while new_xy[1] < 0.0 or new_xy[1] > new_frame.shape[1]:
                    diff /= 2.0
                    new_xy = np.array([a, b]) + diff
                    new_xy = new_xy.astype(np.uint16)

                result = ((a, b), (new_xy[0], new_xy[1]))
                self.lines.append(result)

        # Sample intersections between flow vectors.
        for i, line_a in enumerate(self.lines):
            line_b = self.lines[np.random.randint(0, len(self.lines))]
            int_x, int_y = utils.line_intersection(line_a, line_b)
            intersections[i, :] = int_x, int_y

        intersections = intersections[intersections[:, 0] != 0.0, :]
        FoE = np.median(intersections, axis=0).astype(np.uint16)

        return (FoE[0], FoE[1])

    def check_flow(self, flow_uv: np.ndarray, FoE: Tuple[float, float]) -> np.ndarray:
        if FoE[0] is np.nan:
            return

        result = np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), dtype=np.uint8)

        for y in range(0, flow_uv.shape[0], 10):
            for x in range(0, flow_uv.shape[1], 10):
                # Calculate angle between line from FoE and feature with the flow vector of the feature.
                diff1 = flow_uv[y, x]
                diff2 = np.asarray([x, y]) - np.asarray(FoE)

                flow_magnitude = np.linalg.norm(diff1)
                img_distance = np.linalg.norm(diff2)

                angle_diff = np.abs(np.dot(diff1, diff2) / (flow_magnitude * img_distance))
                color = np.array([255, 255, 255]) * (angle_diff / self.threshold)
                result[y, x, :] = color.astype(np.uint8)

        return result

    def draw(self, frame: np.ndarray, FoE: Tuple[float, float]) -> np.ndarray:
        if FoE[0] is np.nan:
            return

        frame = cv2.circle(frame, (int(FoE[0]), int(FoE[1])), 20, [0, 42, 255], -1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i, line in enumerate(self.lines):
            # Calculate angle between line from FoE and feature with the flow vector of the feature.
            diff1 = np.asarray(line[0]) - np.asarray(line[1])
            diff2 = np.asarray(line[0]) - np.asarray(FoE)

            flow_magnitude = np.linalg.norm(diff1)
            img_distance = np.linalg.norm(diff2)
            flow_angle = np.arctan2(diff1[1], diff1[0]) * 180.0 / np.pi
            img_angle = np.arctan2(diff2[1], diff2[0]) * 180.0 / np.pi

            angle_diff = np.dot(diff1, diff2) / (flow_magnitude * img_distance)
            color = [0, 255, 0]

            if angle_diff < self.threshold:
                color = [0, 0, 255]

            self.mask = cv2.line(self.mask, line[0], line[1], color, 2)
            self.flow_per_distance[i, 0] = img_distance
            self.flow_per_distance[i, 1] = flow_magnitude
            self.flow_per_distance[i, 2] = img_angle
            self.flow_per_distance[i, 3] = flow_angle

        if self.enable_plots:
            plt.figure(1)
            plt.clf()
            plt.xlabel('Distance (pixels)')
            plt.ylabel('Flow magnitude (pixels)')
            plt.grid()
            plt.scatter(self.flow_per_distance[:, 0], self.flow_per_distance[:, 1])

            plt.figure(2)
            plt.clf()
            plt.xlabel('Angle (pixels)')
            plt.ylabel('Flow angle (pixels)')
            plt.grid()
            plt.scatter(self.flow_per_distance[:, 2], self.flow_per_distance[:, 3])
            plt.pause(0.01)

        # Now update the previous frame and previous points
        self.old_gray = frame_gray.copy()

        self.time += 1
        result = cv2.add(frame, self.mask)

        return result
