import utils
import cv2
import numpy as np
from typing import Tuple, cast, Any
from enum import Enum

import im_helpers
from datasets.dataset import Dataset
from lucas_kanade import LucasKanade
from im_helpers import pyramid, sliding_window, get_flow_vis
from frame_result import FrameResult


class Detector:
    class Algorithm(Enum):
        NONE = 0,
        FOE = 1,
        AFFINE = 2,
        HOMOGRAPHY = 3,
        FUNDAMENTAL = 4,
        ESSENTIAL = 5,

    def __init__(self, dataset: Dataset, algorithm: Algorithm = Algorithm.ESSENTIAL, use_sparse_of: bool = False) -> None:
        self.dataset = dataset
        self.algorithm = algorithm
        self.use_sparse_of = use_sparse_of

        flow_width = self.dataset.capture_size[0]
        flow_height = self.dataset.capture_size[1]

        self.sample_size = 1000
        self.border_offset = 20
        self.sample_y = np.random.randint(
            self.border_offset, self.dataset.capture_size[1] - self.border_offset, self.sample_size)
        self.sample_x = np.random.randint(
            self.border_offset, self.dataset.capture_size[0] - self.border_offset, self.sample_size)
        self.coords = np.column_stack((self.sample_x, self.sample_y))

        self.x_coords = np.tile(np.arange(flow_width), (flow_height, 1))
        self.y_coords = np.tile(np.arange(flow_height), (flow_width, 1)).T

        self.history_length = 20
        self.flow_uv_history = np.zeros((self.history_length, flow_height, flow_width, 2))
        self.flow_map_history = np.zeros((self.history_length, flow_height, flow_width))
        self.history_index = 0
        self.confidence: int = 0
        self.use_optimization = False
        self.prev_frame = np.zeros((dataset.capture_size[1], dataset.capture_size[0], 3), dtype=np.uint8)
        self.lucas_kanade = LucasKanade(self.prev_frame)
        self.fov = 90 # degrees
        self.focal_length = 1 / np.tan(np.deg2rad(self.fov) / 2)

    def get_gradient_and_magnitude(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the polar representation of a cartesian flow field.

        Args:
            frame (np.ndarray): flow field (w, h, 2)

        Returns:
            np.ndarray: [description]
        """
        return cast(np.ndarray, np.sqrt(frame[..., 0] ** 2.0 + frame[..., 1] ** 2.0)), \
            cast(np.ndarray, np.arctan2(frame[..., 1], frame[..., 0]))

    def get_rotation(self, flow_uv: np.ndarray, use_fundamental: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # self.get_transformation_matrix()
        R1, R2, t = cv2.decomposeEssentialMat(self.essential)
        return utils.rotation_matrix_to_euler(R1), utils.rotation_matrix_to_euler(R2), t

    def derotate(self, previous_frame_index:int, current_frame_index:int, flow_uv: np.ndarray) -> np.ndarray:
        """Derotate flow field according to IMU data

        Args:
            i (int): frame index
            flow_uv (np.ndarray): (rotated) input flow field

        Returns:
            np.ndarray: derotated flow field
        """
        if current_frame_index < 1:
            return flow_uv

        dt = self.dataset.get_delta_time(current_frame_index)
        w = self.dataset.capture_size[0]
        h = self.dataset.capture_size[1]

        # X displacement corresponds to yaw (Z-axis) and Y displacement corresponds to pitch (Y-axis)
        omega = self.dataset.get_angular_difference(previous_frame_index, current_frame_index) / dt

        x_coords = -(self.x_coords / w - 0.5) * 2.0
        y_coords = -(self.y_coords / h - 0.5) * 2.0

        derotation = np.array(
            [
                +omega[0] * x_coords * y_coords - omega[1] * x_coords ** 2 - omega[1] + omega[2] * y_coords,
                -omega[2] * x_coords + omega[0] + omega[0] * y_coords ** 2 - omega[1] * x_coords * y_coords
            ]
        ).swapaxes(0, 1).swapaxes(1, 2)

        derotation[..., 0] *= w * dt / 2
        derotation[..., 1] *= h * dt / 2

        # idx = 0
        # print('center', np.average(derotation[1024//2, 1920//2, idx]) - np.average(flow_uv[1024//2, 1920//2, idx]))
        # print('topleft', np.average(derotation[0, 0, idx]) - np.average(flow_uv[0, 0, idx]))
        # print('bottomleft', np.average(derotation[-1, 0, idx]) - np.average(flow_uv[-1, 0, idx]))
        # print('bottomright', np.average(derotation[-1, -1, idx]) - np.average(flow_uv[-1, -1, idx]))
        # print('topright', np.average(derotation[0, -1, idx]) - np.average(flow_uv[0, -1, idx]))
        # print(
        #     np.average(im_helpers.get_magnitude(flow_uv - derotation)),
        #     np.std(im_helpers.get_magnitude(flow_uv - derotation)),
        #     np.average(im_helpers.get_magnitude(flow_uv)),
        #     np.average(im_helpers.get_magnitude(flow_uv)) / np.average(im_helpers.get_magnitude(flow_uv - derotation))
        # )
        # print()

        return cast(np.ndarray, flow_uv - derotation)

    def get_transformation_matrix(self, orig_frame: np.ndarray, flow_uv: np.ndarray) -> None:
        """Calculates the affine or homography matrix.

        Args:
            orig_frame (np.ndarray): the input BGR frame
            flow_uv (np.ndarray): the flow field
        """
        coords_old = self.coords
        coords_new = self.coords.astype(np.float64) + \
            flow_uv[self.sample_y, self.sample_x]

        if self.use_sparse_of:
            coords_old_tmp, coords_new_tmp, _ = self.lucas_kanade.get_features(orig_frame)

            if len(coords_old_tmp) > 0 and len(coords_new_tmp) > 0:
                coords_old, coords_new = coords_old_tmp, coords_new_tmp
                self.dataset.logger.info(f'features: {len(coords_new)}')

        if self.algorithm == Detector.Algorithm.HOMOGRAPHY:
            homography, self.confidence = cv2.findHomography(coords_old, coords_new)
            self.homography = np.array(homography)
        elif len(coords_old) > 0 and len(coords_new) > 0:
            if self.algorithm == Detector.Algorithm.AFFINE:
                aff, _ = cv2.estimateAffine2D(coords_old, coords_new)
                self.aff = np.array(aff)
            if self.algorithm == Detector.Algorithm.FUNDAMENTAL:
                fundamental, _ = cv2.findFundamentalMat(coords_old, coords_new, cv2.FM_RANSAC, 0.999, 1.0)
                self.fundamental = np.array(fundamental)
            if self.algorithm == Detector.Algorithm.ESSENTIAL:
                principal_point = (0, 0)
                essential, _ = cv2.findEssentialMat(coords_old, coords_new, self.focal_length,
                    principal_point, cv2.FM_RANSAC, 0.999, 1.0)
                self.essential = np.array(essential)

    def flow_vec_subtract(self, orig_frame: np.ndarray, flow_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculates global motion using perspective or affine matrices and subtracts it from the original field.

        Args:
            orig_frame (np.ndarray): the RGB input frame
            flow_uv (np.ndarray): flow field in cartesian coordinates

        Returns:
            Tuple[np.ndarray, np.ndarray]: the local flow field, its clustered and magnitude version
        """
        self.frame_result: FrameResult = FrameResult()
        global_motion = np.zeros_like(flow_uv)

        # Manual matrix multiplication.
        if self.algorithm == Detector.Algorithm.HOMOGRAPHY:
            global_motion[..., 0] = self.homography[0, 0] * self.x_coords + \
                self.homography[0, 1] * self.y_coords + self.homography[0, 2] - self.x_coords
            global_motion[..., 1] = self.homography[1, 0] * self.x_coords + \
                self.homography[1, 1] * self.y_coords + self.homography[1, 2] - self.y_coords
        else:
            global_motion[..., 0] = self.aff[0, 0] * self.x_coords + \
                self.aff[0, 1] * self.y_coords + self.aff[0, 2] - self.x_coords
            global_motion[..., 1] = self.aff[1, 0] * self.x_coords + \
                self.aff[1, 1] * self.y_coords + self.aff[1, 2] - self.y_coords

        self.flow_uv_warped = global_motion - flow_uv
        flow_uv_warped_vis = get_flow_vis(self.flow_uv_warped)
        self.flow_uv_warped_mag = np.sqrt(self.flow_uv_warped[..., 0] ** 2.0 + self.flow_uv_warped[..., 1] ** 2.0)
        self.flow_max = np.unravel_index(self.flow_uv_warped_mag.argmax(), self.flow_uv_warped_mag.shape)
        # self.cluster_vis, _ = self.clustering(self.flow_uv_warped_mag)
        self.cluster_vis = im_helpers.to_rgb(self.flow_uv_warped_mag)

        flow_uv_warped_mag_vis = im_helpers.to_rgb(self.flow_uv_warped_mag)

        self.opt_window: Tuple[float, utils.Rectangle, np.ndarray, Any] = self.analyze_pyramid(self.cluster_vis)
        window_optimized = self.opt_window[1]
        # self.frame_result.add_box('MAV', 1.0, window_optimized)

        if self.use_optimization:
            window_optimized = self.optimize_window(self.cluster_vis, self.opt_window[1])[1]
            opt_window_list = list(self.opt_window)
            opt_window_list[1] = window_optimized
            self.opt_window = cast(Tuple[float, utils.Rectangle, np.ndarray, Any], opt_window_list)

        for gt in self.dataset.ground_truth:
            self.iou = utils.Rectangle.calculate_iou(window_optimized, gt)

        self.flow_uv_warped_vis = flow_uv_warped_vis
        self.prev_frame = orig_frame
        return flow_uv_warped_vis, self.cluster_vis, flow_uv_warped_mag_vis, get_flow_vis(global_motion)

    def warp_method(self, flow_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Warps the flow field using perspective or affine matrices and subtracts it from the original field.

        Args:
            flow_uv (np.ndarray): flow field in cartesian coordinates

        Returns:
            Tuple[np.ndarray, np.ndarray]: the local flow field and its blockshaped (magnitude) version
        """
        flow_uv = flow_uv.copy()

        if self.algorithm == Detector.Algorithm.HOMOGRAPHY:
            flow_uv_stable = cv2.warpPerspective(flow_uv, self.homography, (752, 480))
        else:
            flow_uv_stable = cv2.warpAffine(flow_uv, self.aff, (752, 480))

        mask = flow_uv_stable[..., :] == np.array([0, 0])
        flow_uv[mask] = flow_uv_stable[mask]
        flow_diff = flow_uv - flow_uv_stable
        self.flow_diff_mag = np.sqrt(flow_diff[..., 0] ** 2.0 + flow_diff[..., 1] ** 2.0)
        flow_max = np.unravel_index(self.flow_diff_mag.argmax(), self.flow_diff_mag.shape)
        flow_diff_vis = get_flow_vis(flow_diff)
        # flow_diff_vis = cv2.circle(
        #     flow_diff_vis, flow_max[::-1], 10, (0, 0, 0), 5)

        blocks = utils.blockshaped(self.flow_diff_mag, 480 // 8, 752 // 8)
        max_mag = np.max(self.flow_diff_mag)
        if max_mag == 0.0:
            max_mag = 1.0

        blocks = np.sum(blocks, 0)
        blocks = blocks / max_mag * 255
        blocks = cv2.resize(blocks, self.flow_diff_mag.shape)
        blocks = np.transpose(blocks)

        blocks_vis = im_helpers.to_rgb(self.flow_diff_mag)
        return flow_diff_vis, blocks_vis

    def draw(self, orig_frame: np.ndarray) -> None:
        """Render ground truths and estimates

        Args:
            orig_frame (np.ndarray): the original frame
        """
        # Plot ground truth.
        for gt in self.dataset.ground_truth:
            orig_frame = cv2.rectangle(
                orig_frame,
                gt.get_topleft_int(),
                gt.get_bottomright_int(),
                (0, 0, 255),
                2
            )

        # self.flow_uv_warped_vis = cv2.circle(
        #     self.flow_uv_warped_vis, self.flow_max[::-1], 10, (0, 0, 0), 5)
        # orig_frame = cv2.circle(
        #     orig_frame, self.flow_max[::-1], 10, (255, 255, 255), 5)

        w: utils.Rectangle = self.opt_window[1]
        # orig_frame = cv2.rectangle(
        #     orig_frame, w.get_topleft(), w.get_bottomright(), (0, 255, 0), 2)
        # self.cluster_vis = cv2.rectangle(
        #     self.cluster_vis, w.get_topleft(), w.get_bottomright(), (0, 255, 0), 2)

        # for gt in self.dataset.ground_truth:
        #     cv2.putText(orig_frame,
        #         f'IoU={self.iou:.02f}',
        #         (gt.get_left(), gt.get_top() - 5),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (255, 0, 0),
        #         2
        #     )


    def analyze_pyramid(self, img: np.ndarray) -> Tuple[float, utils.Rectangle, np.ndarray, Any]:
        """Analyze a frame using pyramid scales.

        Determines which window has heighest score (magnitude).

        Args:
            img (np.ndarray): the image to analyze

        Returns:
            Tuple[float, utils.Rectangle, np.ndarray, float]: score, bounding box, window subimage, maximum flow magnitude
        """
        # Based on: https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
        width, height = (64, 64)
        result: Tuple[float, utils.Rectangle, np.ndarray, Any] = (0, utils.Rectangle((0, 0), (0, 0)), np.zeros(0), 0)

        for resized in pyramid(img, scale=1.5):
            for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(width, height)):
                if window.shape[0] != width or window.shape[1] != height:
                    continue

                score: int = np.sum(window)
                max_flow = np.unravel_index(window.argmax(), window.shape)

                # Check if window has higher score than current maximum.
                if result[0] < score:
                    result = (
                        score,
                        utils.Rectangle((x, y), (window.shape[0], window.shape[1])),
                        window,
                        max_flow
                    )

        return result

    def optimize_window(self, mag_img: np.ndarray, window: utils.Rectangle) -> Tuple[float, utils.Rectangle]:
        """Optimize the window to the area it has to enclose.

        Args:
            mag_img (np.ndarray): the 1-channel image to analyze
            window (utils.Rectangle): initial window estimate

        Returns:
            Tuple[int, utils.Rectangle]: score and optimized window rectangle
        """
        result: Tuple[float, utils.Rectangle] = (0.0, window)
        c = 0

        def get_score(new_window: utils.Rectangle) -> float:
            return float(np.sum(mag_img[int(new_window.get_top()):int(new_window.get_bottom()), int(new_window.get_left()):int(new_window.get_right())]))

        while True:
            window = result[1]
            intermediate_result: Tuple[float, utils.Rectangle] = (0.0, window)

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
                            intermediate_result = (
                                score,
                                new_window
                            )

            if intermediate_result[0] <= result[0]:
                break

            result = intermediate_result
            c += 1

        return result

    def clip(self, img: np.ndarray) -> np.ndarray:
        img[..., 0] = np.clip(img[..., 0], 0, img.shape[1] - 1)
        img[..., 1] = np.clip(img[..., 1], 0, img.shape[0] - 1)
        return img

    def get_history(self, flow_uv: np.ndarray) -> np.ndarray:
        """Calculates the averaged tracked history of flow vectors.

        Args:
            flow_uv (np.ndarray): current flow field

        Returns:
            np.ndarray: history of flow field
        """
        self.flow_uv_history[self.history_index, ...] = flow_uv

        k = (self.history_index + 1) % (self.history_length - 1)
        orig_map: np.ndarray = np.zeros_like(flow_uv, dtype=np.float64)
        orig_map[..., 0] = self.y_coords
        orig_map[..., 1] = self.x_coords
        lookup_map: np.ndarray = np.copy(orig_map)

        while k != (self.history_index) % (self.history_length - 1):
            warped = lookup_map.astype(np.float32)
            lookup_map += cv2.remap(self.flow_uv_history[k, ...], warped[..., 1], warped[..., 0], cv2.INTER_LINEAR)
            k = (k + 1) % self.history_length

        self.history_index = (self.history_index + 1) % self.history_length
        return cast(np.ndarray, lookup_map - orig_map)

    def predict(self, segment: np.ndarray, flow_uv: np.ndarray, orig_frame: np.ndarray) -> None:
        avg = np.average(flow_uv[segment], 0)
        center = self.dataset.ground_truth[0].get_center_int()
        self.prediction = (int(center[0] + avg[0]), int(center[1] + avg[1]))
        orig_frame = cv2.line(orig_frame, center, self.prediction, (0, 0, 255), 5)

    def clustering(self, img: np.ndarray, enable_raw: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Applies k-means clustering to the img.

        Args:
            img (np.ndarray): input grayscale image
            enable_raw (bool, optional): Whether to apply a threshold for visualization. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
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

        if enable_raw:
            return res, np.zeros(0)

        rgb = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        mask = rgb[..., 0] >= 225
        rgb[mask, 1] = 0
        return rgb, mask

    def is_homography_based(self) -> bool:
        return self.algorithm in [
            Detector.Algorithm.HOMOGRAPHY,
        ]
