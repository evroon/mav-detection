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
        self.sample_y = np.random.randint(
            self.border_offset, self.midgard.capture_size[1] - self.border_offset, self.sample_size)
        self.sample_x = np.random.randint(
            self.border_offset, self.midgard.capture_size[0] - self.border_offset, self.sample_size)
        self.coords = np.column_stack((self.sample_x, self.sample_y))

        self.x_coords = np.tile(np.arange(flow_width), (flow_height, 1))
        self.y_coords = np.tile(np.arange(flow_height), (flow_width, 1)).T

        self.ious = np.zeros(self.midgard.N)
        self.history_length = 5
        self.flow_uv_history = np.zeros((self.history_length, flow_height, flow_width, 2))
        self.flow_map_history = np.zeros((self.history_length, flow_height, flow_width))
        self.history_index = 0
        self.use_homography = False

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
        coords_flow = self.coords.astype(
            np.float64) + self.midgard.flow_uv[self.sample_y, self.sample_x]

        if self.use_homography:
            homography, _ = cv2.findHomography(self.coords, coords_flow)
            self.homography = np.array(homography)
        else:
            aff, _ = cv2.estimateAffine2D(self.coords, coords_flow)
            self.aff = np.array(aff)

    def flow_vec_subtract(self):
        flow_uv = self.midgard.flow_uv
        flow_uv_warped = np.zeros_like(flow_uv)

        # Manual matrix multiplication.
        if self.use_homography:
            flow_uv_warped[..., 0] = self.homography[0, 0] * self.x_coords + \
                self.homography[0, 1] * self.y_coords + self.homography[0, 2] - self.x_coords
            flow_uv_warped[..., 1] = self.homography[1, 0] * self.x_coords + \
                self.homography[1, 1] * self.y_coords + self.homography[1, 2] - self.y_coords
        else:
            flow_uv_warped[..., 0] = self.aff[0, 0] * self.x_coords + \
                self.aff[0, 1] * self.y_coords + self.aff[0, 2] - self.x_coords
            flow_uv_warped[..., 1] = self.aff[1, 0] * self.x_coords + \
                self.aff[1, 1] * self.y_coords + self.aff[1, 2] - self.y_coords

        self.flow_uv_warped = flow_uv_warped - flow_uv
        flow_uv_warped_vis = self.midgard.get_flow_vis(self.flow_uv_warped)
        self.flow_uv_warped_mag = np.sqrt(
            self.flow_uv_warped[..., 0] ** 2.0 + self.flow_uv_warped[..., 1] ** 2.0)
        self.flow_max = np.unravel_index(
            self.flow_uv_warped_mag.argmax(), self.flow_uv_warped_mag.shape)

        flow_uv_warped_mag_vis = self.to_rgb(self.flow_uv_warped_mag)

        self.opt_window = self.analyze_pyramid(self.flow_uv_warped_mag)
        self.opt_window[1] = self.optimize_window(self.flow_uv_warped_mag, self.opt_window[1])[1]
        self.iou = utils.Rectangle.calculate_iou(self.opt_window[1], self.midgard.ground_truth) / self.midgard.ground_truth.get_area()

        self.flow_uv_warped_vis = flow_uv_warped_vis
        return flow_uv_warped_vis, flow_uv_warped_mag_vis

    def block_method(self):
        flow_uv = self.midgard.flow_uv.copy()
        flow_uv_stable = cv2.warpAffine(flow_uv, self.aff, (752, 480))
        mask = flow_uv_stable[..., :] == np.array([0, 0])
        flow_uv[mask] = flow_uv_stable[mask]
        flow_diff = flow_uv - flow_uv_stable
        self.flow_diff_mag = np.sqrt(flow_diff[..., 0] ** 2.0 + flow_diff[..., 1] ** 2.0)
        flow_max = np.unravel_index(self.flow_diff_mag.argmax(), self.flow_diff_mag.shape)
        flow_diff_vis = self.midgard.get_flow_vis(flow_diff)
        flow_diff_vis = cv2.circle(
            flow_diff_vis, flow_max[::-1], 10, (0, 0, 0), 5)

        blocks = utils.blockshaped(self.flow_diff_mag, 480 // 8, 752 // 8)
        max_mag = np.max(self.flow_diff_mag)
        if max_mag == 0.0:
            max_mag = 1.0

        blocks = np.sum(blocks, 0)
        blocks = blocks / max_mag * 255
        blocks = cv2.resize(blocks, self.flow_diff_mag.shape)
        blocks = np.transpose(blocks)

        blocks_vis = self.to_rgb(self.flow_diff_mag)
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

        self.flow_uv_warped_vis = cv2.circle(
            self.flow_uv_warped_vis, self.flow_max[::-1], 10, (0, 0, 0), 5)
        self.midgard.orig_frame = cv2.circle(
            self.midgard.orig_frame, self.flow_max[::-1], 10, (255, 255, 255), 5)

        w: utils.Rectangle = self.opt_window[1]
        g = self.midgard.ground_truth
        self.midgard.orig_frame = cv2.rectangle(
            self.midgard.orig_frame, w.get_topleft(), w.get_bottomright(), (0, 255, 0), 2)

        cv2.putText(self.midgard.orig_frame,
            f'IoU={self.iou:.02f}',
            (g.get_left(), g.get_top() - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )


    def analyze_pyramid(self, img: np.ndarray) -> list:
        # Based on: https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
        width, height = (48, 48)
        result = [0, 0, 0, 0]

        for resized in pyramid(img, scale=1.5):
            for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(width, height)):
                if window.shape[0] != width or window.shape[1] != height:
                    continue

                score = np.average(window)
                max_flow = np.unravel_index(window.argmax(), window.shape)

                # Check if window has higher score than current maximum.
                if result[0] < score:
                    result = [
                        score,
                        utils.Rectangle((x, y), window.shape),
                        window,
                        max_flow
                    ]

        return result

    def optimize_window(self, mag_img: np.ndarray, window: utils.Rectangle) -> list:
        result = [0, window]
        c = 0

        def get_score(new_window):
            return np.sum(mag_img[new_window.get_top():new_window.get_bottom(), new_window.get_left():new_window.get_right()]) / new_window.get_area() ** 0.5

        while True:
            window = result[1]
            intermediate_result = [0, None]

            for h in [0, 1]:
                topleft = window.get_topleft()
                bottomright = window.get_bottomright()
                for i in [-1, 1]:
                    for j in [-1, 1]:
                        if h == 0:
                            topleft = (window.get_left() + i, window.get_top() + j)
                        else:
                            bottomright = (window.get_right() + i, window.get_bottom() + j)

                        new_window = utils.Rectangle.from_points(topleft, bottomright)
                        score = get_score(new_window)
                        if score > intermediate_result[0]:
                            intermediate_result = [
                                score,
                                new_window
                            ]

            if intermediate_result[0] <= result[0]:
                break

            result = intermediate_result
            c += 1

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
    #         feature_pos = np.array(ground_truth.get_center())
    #         feature_pos = np.clip(feature_pos, min_coords, max_coords)

    def get_magnitude(self, img: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(img ** 2.0, axis=-1))

    def to_rgb(self, img: np.ndarray)-> np.ndarray:
        max_intensity = np.max(img)
        if max_intensity == 0.0:
            max_intensity = 1.0

        return cv2.cvtColor(np.around(img * 255 / max_intensity).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def get_history(self) -> np.ndarray:
        self.flow_uv_history[self.history_index, ...] = self.midgard.flow_uv
        self.flow_map_history[self.history_index, ...] = self.flow_uv_warped_mag
        k = (self.history_index + 2) % (self.history_length - 1)
        summed_mag = self.flow_map_history[(self.history_index + 1) % (self.history_length - 1), ...]

        while k != (self.history_index  + 1) % (self.history_length - 1):
            lookup_x = self.x_coords + self.flow_uv_history[k, ..., 0]
            lookup_y = self.y_coords + self.flow_uv_history[k, ..., 1]
            lookup_x = np.around(np.clip(lookup_x, 0.0, summed_mag.shape[1] - 1)).astype(np.uint16)
            lookup_y = np.around(np.clip(lookup_y, 0.0, summed_mag.shape[0] - 1)).astype(np.uint16)

            summed_mag = summed_mag[lookup_y, lookup_x]
            summed_mag += self.flow_map_history[(k - 1) % (self.history_length - 1), ...]

            k = (k + 1) % self.history_length

        self.history_index = (self.history_index + 1) % self.history_length
        return self.to_rgb(summed_mag)

    def predict(self, segment: np.ndarray):
        avg = np.average(self.midgard.flow_uv[segment], 0)
        center = self.midgard.ground_truth.get_center_int()
        self.prediction = (int(center[0] + avg[0]), int(center[1] + avg[1]))
        self.midgard.orig_frame = cv2.line(self.midgard.orig_frame, center, self.prediction, (0, 0, 255), 5)

    def clustering(self, img):
        K = 8
        Z = img.reshape((-1, 3)).astype(np.float32)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        max_mag = np.max(center)
        if max_mag == 0.0:
            max_mag = 1.0

        center = np.uint8(center * 255 / max_mag)
        res = center[label.flatten()]
        res = res.reshape((img.shape))

        rgb = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        mask = rgb[..., 0] >= 225
        rgb[mask, 1] = 0
        return rgb, mask
