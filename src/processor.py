import utils
import cv2
import os
import numpy as np
import glob
import shutil
import subprocess
import json

import im_helpers
from typing import List, Dict, Tuple, Optional
from detector import Detector
from run_config import RunConfig
from frame_result import FrameResult
from focus_of_expansion import FocusOfExpansion

class Processor:
    '''Converts dataset for use with YOLOv4 or acts as detector.'''
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.logger = self.config.logger
        self.sequence = config.sequence
        self.debug_mode = config.debug
        self.headless = config.headless
        self.dataset = config.get_dataset()
        self.detector = Detector(self.dataset)
        self.detection_results: Dict[int, FrameResult] = dict()
        self.output: Optional[cv2.VideoWriter] = None
        self.use_gt_of = False
        self.frame_step_size = 1

        self.frame_index, self.start_frame = 0, 100
        self.is_exiting = False

        self.midgard_path = os.environ['MIDGARD_PATH']
        self.focus_of_expansion = FocusOfExpansion(self.detector.lucas_kanade)
        self.old_frame: np.ndarray = np.zeros((self.dataset.capture_size[1], self.dataset.capture_size[0], 3), dtype=np.uint8)
        self.colorbar = im_helpers.plot_colorbar()

    def annotation_to_yolo(self, rects: List[utils.Rectangle]) -> str:
        """Converts the rectangles to the text format read by YOLOv4

        Args:
            rects (list): the input rectangles

        Returns:
            str: a list of bounding boxes in format '0 {center_x} {center_y} {size_x} {size_y}' with coordinates in range [0, 1]
        """
        result: str = ''
        for rect in rects:
            result += rect.to_yolo(self.dataset.resolution)

        return result

    def is_active(self) -> bool:
        """Returns whether the process is still active"""
        return self.frame_index < self.dataset.N - 1 and not self.is_exiting

    def write(self, frame: np.ndarray) -> None:
        """Writes the frame to the disk

        Shows the frame as well if debug_mode is True.

        Args:
            frame (np.ndarray): the final frame to save
        """
        output_path = f'{self.dataset.seq_path}/processed.mp4'
        if self.output is None:
            height, width, channels = frame.shape
            self.output = utils.get_output(output_path, capture_size=(width, height), is_grey=(channels==1))
            self.logger.info(f'Writing output to: {output_path}')

        self.output.write(frame)

        if not self.headless:
            cv2.imshow('output', frame)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                self.is_exiting = True

        with open(f'{self.dataset.results_path}/image_{self.frame_index:05d}.json', 'w') as f:
            f.write(json.dumps(utils.get_json(self.config.results[self.frame_index]), indent=4, sort_keys=True))

        self.frame_index += self.frame_step_size

        if self.frame_index % int(self.dataset.N / 10) == 0:
            self.logger.info(f'{self.frame_index / self.dataset.N * 100:.2f}% {self.frame_index} / {self.dataset.N}')

    def remove_contents_of_folder(self, folder: str) -> None:
        """Remove all content of a directory

        Args:
            folder (string): the directory to delete
        """
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def process_image(self, src: str, dst: str) -> None:
        """Processes an image of the dataset and places it in the target directory

        Args:
            src (str): source image path
            dst (str): destination image path
        """
        if self.mode == RunConfig.Mode.APPEARANCE_RGB:
            shutil.copy2(src, dst)
        else:
            img = cv2.imread(src)

            if img.shape != self.dataset.capture_shape:
                img = cv2.resize(img, self.dataset.capture_shape)

            if self.mode == RunConfig.Mode.FLOW_UV:
                flow_uv = self.dataset.get_flow_uv(self.frame_index)
                flow_vis = im_helpers.get_flow_vis(flow_uv)
                cv2.imwrite(dst, flow_vis)
            elif self.mode in [RunConfig.Mode.FLOW_FOE_CLUSTERING, RunConfig.Mode.FLOW_FOE_YOLO]:
                orig_frame = self.dataset.get_frame()
                flow_uv = self.dataset.get_flow_uv(self.frame_index)
                self.detector.get_transformation_matrix(orig_frame, flow_uv)
                self.detector.flow_vec_subtract(orig_frame, flow_uv)
                cv2.imwrite(dst, self.detector.flow_uv_warped_mag * 255 / np.max(self.detector.flow_uv_warped_mag))

    def process_annot(self, src: str, dst: str) -> None:
        """Processes an annotation file of the dataset and places it in the target directory

        Args:
            src (str): source annotation file path
            dst (str): destination annotation file path
        """
        shutil.copy2(src, dst)

    def get_data(self, sequence: str, with_yolo_ann: bool = True) -> Tuple[List[str], List[str]]:
        self.img_path = f'{self.midgard_path}/{sequence}/images'
        self.ann_path = f'{self.midgard_path}/{sequence}/annotation'
        cal_glob =  glob.glob(f'{self.midgard_path}/{sequence}/info/calibration/*.txt')
        self.cal_path = ''

        if len(cal_glob) > 0:
            self.cal_path = cal_glob[0]

        images = utils.sorted_glob(f'{self.img_path}/image_*.png')
        ann_extension = 'txt' if with_yolo_ann else 'csv'
        annotations = utils.sorted_glob(f'{self.ann_path}/*.{ann_extension}')
        return images, annotations

    def annotations_to_yolo(self) -> None:
        """Converts annotations from MIDGARD to YOLOv4 format."""
        sequences = self.config.get_all_sequences()

        for sequence in sequences:
            self.logger.info(f'Converting annotations to YOLOv4 format for sequence: {sequence}')
            _, annotations = self.get_data(sequence, False)

            # Remove existing .txt annotation files.
            for file in glob.glob(f'{self.ann_path}/*.txt'):
                os.remove(file)

            for ann_src in annotations:
                output_path = ann_src.replace('annot_', 'image_').replace('csv', 'txt')

                with open(output_path, 'w') as f:
                    ann = self.dataset.get_annotation(self.frame_index, ann_src)
                    f.writelines(self.annotation_to_yolo(ann))

                self.frame_index += 1

    def prepare_sequence(self, sequence: str) -> None:
        """Prepare a sequence of the dataset

        Args:
            sequence (str): which sequence to prepare, for example 'indoor-modern/sports-hall'
        """
        self.logger.info(f'Preparing sequence {sequence}')
        self.sequence = sequence
        self.flo_path = f'{self.img_path}/output/inference/run.epoch-0-flow-field'

        self.capture_shape = self.dataset.get_capture_shape()
        self.resolution = np.array(self.capture_shape)[:2][::-1]

        images, annotations = self.get_data(sequence)
        flow_fields = glob.glob(f'{self.flo_path}/*.flo')

        self.frame_index = 0
        self.N = len(images)

        if len(images) != len(annotations) or len(images) - 1 != len(flow_fields):
            print('Input counts: (images, annotations, flow fields):', len(images), len(annotations), len(flow_fields))
            raise ValueError('Input sizes do not match.')

        for img_src, ann_src in zip(images, annotations):
            # Skip the last frame for optical flow inputs, as it does not exist.
            if not (self.mode != RunConfig.Mode.APPEARANCE_RGB and self.frame_index >= self.N - 2):
                self.process_image(img_src, f'{self.img_dest_path}/{self.output_index:06d}.png')
                self.process_annot(ann_src, f'{self.ann_dest_path}/{self.output_index:06d}.txt')
                self.output_index += 1
                self.frame_index += 1

    def convert(self, mode: RunConfig.Mode) -> None:
        """Processes the dataset"""

        # The number of channels per mode.
        channel_options = {
            RunConfig.Mode.APPEARANCE_RGB: 3,
            RunConfig.Mode.FLOW_UV: 2,
            RunConfig.Mode.FLOW_RADIAL: 1,
            RunConfig.Mode.FLOW_FOE_YOLO: 1,
            RunConfig.Mode.FLOW_FOE_CLUSTERING: 1,
        }

        self.dest_path = os.environ['YOLOv4_PATH'] + '/dataset'
        self.img_dest_path = f'{self.dest_path}/images'
        self.ann_dest_path = f'{self.dest_path}/labels/yolo'

        sequences = self.config.settings['train_sequences']

        self.mode = mode
        self.channels = channel_options[self.mode]
        self.output_index = 0

        self.remove_contents_of_folder(self.img_dest_path)
        self.remove_contents_of_folder(self.ann_dest_path)

        print(f'Mode: {self.mode}')

        for sequence in sequences:
            self.prepare_sequence(sequence)

    def undistort(self) -> None:
        """Undistorts the MIDGARD dataset."""
        sequences = self.config.get_all_sequences()

        for sequence in sequences:
            images, _ = self.get_data(sequence)
            undistort_exec = os.environ['UNDISTORT_PATH']
            output_dir = img_out = os.path.dirname(self.img_path) + '/undistorted'

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            if self.cal_path == '':
                continue

            for img in images:
                img_out = f'{output_dir}/{os.path.basename(img)}'

                if os.path.exists(img_out):
                    continue

                print(f'Undistorting: {img_out}')

                command = [undistort_exec, '--run', self.cal_path, img, img_out]

                with open(os.devnull, 'w') as devnull:
                    subprocess.call(command, stdout=devnull)

    def run_detection(self) -> Dict[int, FrameResult]:
        """Runs the detection."""

        im_helpers.get_colorwheel()
        prev_bbox_left = 0

        while self.is_active():
            orig_frame = self.dataset.get_frame()

            if self.detector.is_homography_based():
                self.flow_uv = self.dataset.get_flow_uv(self.frame_index)
                self.flow_vis = im_helpers.get_flow_vis(self.flow_uv)
                self.dataset.get_annotation(self.frame_index)

                self.detector.get_transformation_matrix(orig_frame, self.flow_uv)
                flow_uv_warped_vis, cluster_vis, _, global_motion_vis = self.detector.flow_vec_subtract(orig_frame, self.flow_uv)
                global_motion_vis = global_motion_vis.astype(np.uint8)

                if self.debug_mode:
                    self.detector.draw(orig_frame)

                    top_frames = np.hstack((orig_frame, global_motion_vis, flow_uv_warped_vis))
                    bottom_frames = np.hstack((self.flow_vis, global_motion_vis, cluster_vis))
                    self.write(np.vstack((top_frames, bottom_frames)))
                    self.detection_results[self.frame_index] = self.detector.frame_result
                else:
                    self.write(cluster_vis)
            else:
                if self.use_gt_of:
                    self.flow_uv = self.dataset.get_gt_of(self.frame_index)
                else:
                    self.flow_uv = self.dataset.get_flow_uv(self.frame_index)

                if self.flow_uv is None:
                    raise ValueError('Could not load flow field.')

                mask = self.dataset.get_sky_segmentation(self.frame_index)
                sky_tpr, sky_fpr = self.dataset.validate_sky_segment(mask, self.dataset.get_depth(self.frame_index))

                self.flow_vis = im_helpers.get_flow_vis(self.flow_uv)
                self.flow_uv_derotated = self.detector.derotate(self.frame_index - self.frame_step_size, self.frame_index, self.flow_uv)
                self.flow_mag = im_helpers.get_magnitude(self.flow_uv_derotated)

                FoE_dense  = self.focus_of_expansion.get_FOE_dense(self.flow_uv_derotated)
                FoE_gt = self.dataset.get_gt_foe(self.frame_index)
                FoE: Tuple[float, float] = utils.assert_type(FoE_dense)

                phi_angle = self.focus_of_expansion.check_flow(self.flow_uv_derotated, FoE)
                phi_angle_rgb = im_helpers.to_rgb(phi_angle, max_value=180.0)
                result_img = im_helpers.apply_colormap(phi_angle_rgb, max_value=180.0)

                frameresult = FrameResult()
                frameresult.foe_dense = FoE_dense
                frameresult.foe_gt = utils.assert_type(FoE_gt)

                if True:
                    segmentation = self.dataset.get_segmentation(self.frame_index)[..., 0]
                    estimate = phi_angle * (mask != True) * (self.flow_mag > 1.0)
                    angle_threshold = 20

                    drone_flow_avg = np.average(self.flow_uv[segmentation > 127], axis=0)
                    drone_flow_avg_gt = np.average(self.dataset.get_gt_of(self.frame_index)[segmentation > 127], axis=0)

                    bounding_box = im_helpers.get_simple_bounding_box(segmentation)

                    print(bounding_box.get_left() - prev_bbox_left, drone_flow_avg[0], drone_flow_avg_gt[0], 1920 / self.dataset.N)
                    prev_bbox_left = bounding_box.get_left()

                    tpr, fpr = im_helpers.calculate_tpr_fpr(segmentation, 255 * (estimate > angle_threshold))
                    frameresult.tpr = tpr
                    frameresult.fpr = fpr
                    frameresult.sky_tpr = sky_tpr
                    frameresult.sky_fpr = sky_fpr
                    frameresult.drone_flow_pixels = (drone_flow_avg_gt[0], drone_flow_avg_gt[1])
                    frameresult.drone_size_pixels = np.sum(segmentation > 127)

                    utils.create_if_not_exists(self.dataset.result_imgs_path)
                    img = im_helpers.apply_colormap(estimate, max_value=180)
                    img = self.focus_of_expansion.draw_FoE(img, FoE_dense,  [255, 255, 255])

                    cv2.imwrite(f'{self.dataset.result_imgs_path}/image_{self.frame_index:05d}.png', img)

                for img in [orig_frame, result_img]:
                    img = self.focus_of_expansion.draw_FoE(img, FoE_dense,  [0, 255, 0])

                    if FoE_gt is not None:
                        img = self.focus_of_expansion.draw_FoE(img, FoE_gt, [255, 255, 255])

                self.detection_results[self.frame_index] = frameresult
                self.config.results[self.frame_index] = frameresult

                if result_img is not None and np.sum(result_img) > 0:
                    mask_vis = orig_frame
                    mask_vis[estimate > angle_threshold, :] = mask_vis[estimate > angle_threshold, :] * 0.5 + 127
                    self.write(mask_vis)
                else:
                    print('An error occured while processing frames.')

        return self.detection_results

    def release(self) -> None:
        """Release all media resources"""
        self.dataset.release()
        if self.output is not None:
            self.output.release()
