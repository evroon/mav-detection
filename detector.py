import utils
import cv2
import os
import numpy as np
import flow_vis
from midgard import Midgard
from im_helpers import pyramid
from im_helpers import sliding_window


class Detector:
    def __init__(self, midgard: Midgard):
        self.midgard = midgard

        flow_width = self.midgard.capture_size[0]
        flow_height = self.midgard.capture_size[1]

        self.sample_size = 1000
        self.border_offset = 20
        self.sample_y = np.random.randint(self.border_offset, self.midgard.capture_size[1] - self.border_offset, self.sample_size)
        self.sample_x = np.random.randint(self.border_offset, self.midgard.capture_size[0] - self.border_offset, self.sample_size)
        self.coords = np.column_stack((self.sample_x, self.sample_y))

        self.x_coords = np.tile(np.arange(flow_width), (flow_height, 1))
        self.y_coords = np.tile(np.arange(flow_height), (flow_width, 1)).T

        # feature_pos = np.array([220.0, 280.0])
        # min_coords = np.zeros(2)
        # max_coords = self.midgard.capture_size[::-1] - np.ones(2)

    def get_gradient_and_magnitude(self, frame: np.ndarray) -> np.ndarray:
        """Returns the polar representation of a cartesian flow field.

        Args:
            frame (np.ndarray): flow field (w, h, 2)

        Returns:
            np.ndarray: [description]
        """
        return np.sqrt(frame[..., 0] ** 2.0 + frame[..., 1] ** 2.0), \
               np.arctan2(frame[..., 1], frame[..., 0])

    def get_fft(self, frame):
        fft = np.fft.fft2(frame[..., 0])
        fshift = np.fft.fftshift(fft)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        magnitude_rgb = np.zeros_like(frame)
        magnitude_rgb[..., 0] = magnitude_spectrum
        return magnitude_rgb

    def get_affine_matrix(self):
        coords_flow = self.coords.astype(np.float64) + self.midgard.flow_uv[self.sample_y, self.sample_x]
        aff, _ = cv2.estimateAffine2D(self.coords, coords_flow)
        self.aff = np.array(aff)

    def flow_vec_subtract(self):
        flow_uv = self.midgard.flow_uv
        flow_uv_warped = np.zeros_like(flow_uv)

        # Manual matrix multiplication.
        flow_uv_warped[..., 0] = self.aff[0, 0] * self.x_coords + self.aff[0, 1] * self.y_coords + self.aff[0, 2] - self.x_coords
        flow_uv_warped[..., 1] = self.aff[1, 0] * self.x_coords + self.aff[1, 1] * self.y_coords + self.aff[1, 2] - self.y_coords

        flow_uv_warped = flow_uv_warped - flow_uv
        flow_uv_warped_vis = self.midgard.get_flow_vis(flow_uv_warped)
        flow_uv_warped_mag = np.sqrt(flow_uv_warped[..., 0] ** 2.0 + flow_uv_warped[..., 1] ** 2.0)
        flow_uv_warped_mag_vis = np.zeros_like(self.midgard.flow_vis)
        self.flow_max = np.unravel_index(flow_uv_warped_mag.argmax(), flow_uv_warped_mag.shape)

        for i in range(3):
            flow_uv_warped_mag_vis[..., i] = flow_uv_warped_mag / np.max(flow_uv_warped_mag) * 255

        self.opt_window = self.analyze_pyramid(flow_uv_warped_mag_vis)

        self.flow_uv_warped_vis = flow_uv_warped_vis
        return flow_uv_warped_vis, flow_uv_warped_mag_vis

    def block_method(self):
        flow_uv = self.midgard.flow_uv
        flow_uv_stable = cv2.warpAffine(flow_uv, self.aff, (752, 480))
        mask = flow_uv_stable[..., :] == np.array([0, 0])
        flow_uv[mask] = flow_uv_stable[mask]
        flow_diff = flow_uv - flow_uv_stable
        flow_diff_mag = np.sqrt(flow_diff[..., 0] ** 2.0 + flow_diff[..., 1] ** 2.0)
        flow_max = np.unravel_index(flow_diff_mag.argmax(), flow_diff_mag.shape)
        flow_diff_vis = self.midgard.get_flow_vis(flow_diff)
        flow_diff_vis = cv2.circle(flow_diff_vis, flow_max[::-1], 10, (0, 0, 0), 5)

        blocks = utils.blockshaped(flow_diff_mag, 480 // 8, 752 // 8)
        blocks = np.sum(blocks, 0)
        blocks = blocks / np.max(blocks) * 255
        blocks = cv2.resize(blocks, flow_diff_mag.shape)
        blocks = np.transpose(blocks)

        blocks_vis = np.zeros((flow_diff_mag.shape[0], flow_diff_mag.shape[1], 3), dtype=np.uint8)
        for i in range(3):
            blocks_vis[..., i] = flow_diff_mag / np.max(flow_diff_mag) * 255

        return flow_diff_vis, blocks_vis

    def draw(self):
        # Plot ground truth.
        self.midgard.orig_frame = cv2.rectangle(
            self.midgard.orig_frame,
            self.midgard.ground_truth.get_topleft(),
            self.midgard.ground_truth.get_bottomright(),
            (0, 0, 255),
            3
        )

        self.flow_uv_warped_vis = cv2.circle(self.flow_uv_warped_vis, self.flow_max[::-1], 10, (0, 0, 0), 5)
        self.midgard.orig_frame = cv2.circle(self.midgard.orig_frame, self.flow_max[::-1], 10, (255, 255, 255), 5)

        w = self.opt_window
        self.midgard.orig_frame = cv2.rectangle(self.midgard.orig_frame, (w[1], w[2]), (w[1] + w[3].shape[1], w[2] + w[3].shape[0]), (0, 255, 0), 2)

    def analyze_pyramid(self, img):
        # Based on: https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
        width, height = (48, 48)
        result = [0, 0, 0, 0]

        for resized in pyramid(img, scale=1.5):
            for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(width, height)):
                if window.shape[0] != width or window.shape[1] != height:
                    continue

                # Check if window has higher score.
                score = np.average(window)
                if result[0] < score:
                    result = [
                        score,
                        x, y,
                        window
                    ]

        return result

    # def get_histogram(self):
    #     magnitude, gradient = midgard.get_gradient_and_magnitude(flow_uv)

    #     mag_hist, mag_edges = np.histogram(magnitude, 10)
    #     gra_hist, gra_edges = np.histogram(gradient,  10)
    #     mag_max_id = np.argmax(mag_hist)
    #     gra_max_id = np.argmax(gra_hist)
    #     mag_range = mag_edges[mag_max_id], mag_edges[mag_max_id + 1]
    #     gra_range = gra_edges[gra_max_id], gra_edges[gra_max_id + 1]
    #     ones = np.ones_like(magnitude)

    #     feature_pos_int = feature_pos.astype(np.uint32).tolist()
    #     feature_pos += flow_uv[feature_pos_int[0], feature_pos_int[1], :]
    #     feature_pos_clipped = np.clip(feature_pos, min_coords, max_coords)

    #     if feature_pos[0] != feature_pos_clipped[0] or feature_pos[1] != feature_pos_clipped[1]:
    #         feature_pos = np.array(ground_truth.get_center())[::-1]
    #         feature_pos = np.clip(feature_pos, min_coords, max_coords)

    # def sample_box(self):
    #     res2 = np.zeros_like(orig_frame)

    #     def clip_axis(axis, value):
    #         return min(res2.shape[axis] - 1, max(0, value))

    #     left    = clip_axis(1, ground_truth.get_left())
    #     right   = clip_axis(1, ground_truth.get_right())
    #     top     = clip_axis(0, ground_truth.get_top() - 15)
    #     bottom  = clip_axis(0, ground_truth.get_bottom() - 15)

    #     x_range = np.arange(left, right, 1)
    #     y_range = np.arange(top, bottom, 1)
    #     reference = flow_vis[int(y_range[len(y_range)//2]), int(x_range[len(x_range)//2]), :]
    #     orig_frame = cv2.circle(orig_frame, (int(x_range[len(x_range)//2]), int(y_range[len(y_range)//2])), 10, (0, 255, 0))

    #     for x in x_range:
    #         for y in y_range:
    #             if np.sum(flow_vis[y, x, :] - reference) < 40:
    #                 res2[y, x, :] = 255

    # def clustering(self):
    #     mask = (magnitude > ones * mag_range[0]) & (magnitude < ones * mag_range[1])
    #     res2[mask, 0] = 255

    #     Z = flow_vis.reshape((-1, 3))
    #     # convert to np.float32
    #     Z = np.float32(Z)
    #     # define criteria, number of clusters(K) and apply kmeans()
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #     K = 8
    #     ret, label, center = cv2.kmeans(
    #         Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #     # Now convert back into uint8, and make original image
    #     center = np.uint8(center)
    #     res = center[label.flatten()]
    #     res2 = res.reshape((flow_vis.shape))
