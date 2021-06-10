import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
from typing import Tuple, List, cast

from lucas_kanade import LucasKanade
import im_helpers

plt.rcParams['axes.axisbelow'] = True


class FocusOfExpansion:
    def __init__(self, lucas_kanade: LucasKanade) -> None:
        self.lucas_kanade = lucas_kanade
        self.time = 0
        self.roll_back = 20
        self.num_features = 0
        self.enable_plots = False
        self.max_flow = 0.0 # maximum flow in the image (degrees).
        self.radial_threshold = np.cos(np.deg2rad(15))
        self.magnitude_threshold = 2.5
        self.ransac_threshold = 30.0 # pixels
        self.color = np.random.randint(0, 255, (self.lucas_kanade.total_num_corners, 3))
        self.trace = np.zeros((self.lucas_kanade.total_num_corners, 2000), dtype=np.int)
        self.random_lines = np.random.randint(0, self.lucas_kanade.total_num_corners, self.lucas_kanade.total_num_corners)
        flow_height, flow_width = self.lucas_kanade.old_frame.shape[0], self.lucas_kanade.old_frame.shape[1]

        self.x_coords = np.tile(np.arange(flow_width), (flow_height, 1))
        self.y_coords = np.tile(np.arange(flow_height), (flow_width, 1)).T

    def ransac(self, estimates: np.ndarray) -> Tuple[float, float]:
        """Simple RANSAC scheme to calculate the FoE position

        Args:
            estimates (np.ndarray): estimate FoE positions/intersections

        Returns:
            Tuple[float, float]: FoE location estimation
        """
        optimum = 0
        optimal_foe = (0.0, 0.0)

        for i in range(estimates.shape[0]):
            chosen_sample = estimates[i]
            count = im_helpers.get_magnitude(estimates - chosen_sample)
            inliers = count[count < self.ransac_threshold]
            score = inliers.shape[0] - 1

            if score > optimum:
                optimum = score
                optimal_foe = cast(Tuple[float, float], tuple(chosen_sample))

        return optimal_foe

    def get_FOE_dense(self, flow_uv: np.ndarray) -> Tuple[float, float]:
        """Get FoE location using dense optical flow.

        Args:
            flow_uv (np.ndarray): current optical flow field

        Returns:
            Tuple[float, float]: FoE estimation location
        """
        N = 1000
        intersections = np.zeros((N, 2))

        # rand1 has shape (2*N, (y, x)).
        rand1 = np.zeros((N*2, 2), dtype=np.uint32)
        rand1[..., 0] = np.random.randint(0, flow_uv.shape[0], N*2)
        rand1[..., 1] = np.random.randint(0, flow_uv.shape[1], N*2)

        # Sample intersections between flow vectors.
        for i in range(N):
            coord1, coord2 = rand1[i, :], rand1[i+N, :]
            flow1, flow2 = flow_uv[coord1[0], coord1[1], :], flow_uv[coord2[0], coord2[1], :]

            if im_helpers.get_magnitude(flow2) < self.magnitude_threshold:
                continue

            coord1 = coord1[::-1]
            coord2 = coord2[::-1]
            intersections[i, :] = utils.line_intersection((coord1, flow1 + coord1), (coord2, flow2 + coord2))

        intersections = intersections[intersections[:, 0] != 0.0, :]
        return self.ransac(intersections)

    def get_FOE_sparse(self, old_frame: np.ndarray, new_frame: np.ndarray) -> Tuple[float, float]:
        """Get FoE location using sparse optical flow.

        Args:
            old_frame (np.ndarray): previous frame
            new_frame (np.ndarray): current frame

        Returns:
            Tuple[float, float]: FoE estimation location
        """
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

            l = self.trace[i, 0] + 1
            self.trace[i, l] = c
            self.trace[i, l+1] = d
            self.trace[i, 0] += 2
            color = self.color[i].tolist()
            self.num_features += 1

            if l >= 3:
                k = 1 if l < 1 + self.roll_back * 2 else l - self.roll_back * 2
                a, b = self.trace[i, l:l+2]
                c, d = self.trace[i, k:k+2]

                diff = np.array([float(c) - float(a), float(d) - float(b)])

                new_xy = np.array([a, b]) + diff
                new_xy = new_xy.astype(np.uint16)

                # Let flow vector fit inside image bounds.
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
        return self.ransac(intersections)

    def get_phi(self, derotated_flow_uv: np.ndarray, FoE: Tuple[float, float]) -> np.ndarray:
        """Calculates the angle phi per pixel.

        Args:
            derotated_flow_uv (np.ndarray): the derotated flow field
            FoE (Tuple[float, float]): the Focus of Expansion

        Returns:
            np.ndarray: BGR image with higher intensities for higher local motion.
        """
        if FoE[0] is np.nan:
            return

        # Calculate angle between line from FoE and feature with the flow vector of the feature.
        diff1 = derotated_flow_uv
        diff2 = np.zeros_like(derotated_flow_uv)
        diff2[..., 0] = self.x_coords - FoE[0]
        diff2[..., 1] = self.y_coords - FoE[1]

        flow_magnitude = im_helpers.get_magnitude(diff1)
        img_distance = im_helpers.get_magnitude(diff2)
        norm = np.maximum(np.ones_like(flow_magnitude) * 1e-6, flow_magnitude * img_distance)

        arccos_arg = (diff1[..., 0] * diff2[..., 0] + diff1[..., 1] * diff2[..., 1]) / norm
        arccos_arg = np.clip(arccos_arg, -1, 1)
        angle_diff = np.arccos(arccos_arg)

        angle_diff[np.isnan(angle_diff)] = 0
        phi_angle_deg = np.rad2deg(angle_diff)
        self.max_flow = np.max(phi_angle_deg)

        # mask = im_helpers.get_magnitude(derotated_flow_uv) < self.magnitude_threshold
        # phi_angle_deg[mask] = 0

        return phi_angle_deg

    def draw_FoE(self, frame: np.ndarray, FoE: Tuple[float, float], color: List[int]=[0, 42, 255], radius: float = 10) -> np.ndarray:
        """Draw Focus of Expansion circle in an image

        Args:
            frame (np.ndarray): the image to draw on
            FoE (Tuple[float, float]): Location of the FoE
            color (List[int], optional): color of the circle in BGR space. Defaults to [0, 42, 255].
            radius (float): Radius of the circle to draw

        Returns:
            np.ndarray: resulting image
        """
        if FoE[0] is np.nan or FoE[1] is np.nan or np.abs(FoE[0]) > 1e9 or np.abs(FoE[1]) > 1e9:
            return frame

        return cv2.circle(frame, (int(FoE[0]), int(FoE[1])), radius, color, -1)

    def draw(self, frame: np.ndarray, FoE: Tuple[float, float]) -> np.ndarray:
        """Draw FoE algorithm visualization

        Args:
            frame (np.ndarray): current frame
            FoE (Tuple[float, float]): estimated FoE location

        Returns:
            np.ndarray: angle difference between vector towards FoE and flow vector
        """
        if FoE[0] is np.nan:
            return

        frame = cv2.circle(frame, (int(FoE[0]), int(FoE[1])), 20, [0, 42, 255], -1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i, line in enumerate(self.lines):
            # Calculate angle between line from FoE and feature with the flow vector of the feature.
            diff1 = np.asarray(line[0]) - np.asarray(line[1])
            diff2 = np.asarray(line[0]) - np.asarray(FoE)

            flow_magnitude = im_helpers.get_magnitude(diff1)
            img_distance = im_helpers.get_magnitude(diff2)
            flow_angle = np.arctan2(diff1[1], diff1[0]) * 180.0 / np.pi
            img_angle = np.arctan2(diff2[1], diff2[0]) * 180.0 / np.pi

            angle_diff = np.dot(diff1, diff2) / (flow_magnitude * img_distance)
            color = [0, 255, 0]

            if angle_diff < self.radial_threshold:
                color = [0, 0, 255]

            self.mask = cv2.line(self.mask, line[0], line[1], color, 2)

        # Now update the previous frame and previous points
        self.old_gray = frame_gray.copy()

        self.time += 1
        return cv2.add(frame, self.mask)
