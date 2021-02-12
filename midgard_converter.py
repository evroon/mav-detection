import utils
import cv2
import os
import numpy as np
import flow_vis
from enum import Enum
import glob
import shutil
from im_helpers import get_flow_radial, get_flow_vis
from typing import Optional, List
from midgard import Midgard
from detector import Detector

class MidgardConverter:
    '''Converts MIDGARD dataset for use with YOLOv4.'''
    def __init__(self, sequence: str, debug_mode: bool, headless: bool) -> None:
        self.sequence = sequence
        self.debug_mode = debug_mode
        self.headless = headless
        self.midgard = Midgard(sequence)
        self.detector = Detector(self.midgard)
        self.frame_columns, self.frame_rows = 1, 1

        if debug_mode:
            self.frame_columns, self.frame_rows = 4, 2

        output_size = (self.midgard.capture_size[0] * self.frame_columns, self.midgard.capture_size[1] * self.frame_rows)
        self.output = utils.get_output('detection', capture_size=output_size, is_grey=not debug_mode)

        self.frame_index, self.start_frame = 0, 100
        self.is_exiting = False

    def annotation_to_yolo(self, rects: List[utils.Rectangle]) -> str:
        """Converts the rectangles to the text format read by YOLOv4

        Args:
            rects (list): the input rectangles

        Returns:
            str: a list of bounding boxes in format '0 {center_x} {center_y} {size_x} {size_y}' with coordinates in range [0, 1]
        """
        result: str = ''
        for rect in rects:
            center = np.array(rect.get_center()) / self.midgard.resolution.astype(np.float)
            size = np.array(rect.size) / (2.0 * self.midgard.resolution.astype(np.float))
            result += f'0 {center[0]} {center[1]} {size[0]} {size[1]}\n'

        return result

    def is_active(self) -> bool:
        """Returns whether the process is still active"""
        return self.frame_index < self.midgard.N - 1 and not self.is_exiting

    def write(self, frame: np.ndarray) -> None:
        """Writes the frame to the disk

        Shows the frame as well if debug_mode is True.

        Args:
            frame (np.ndarray): the final frame to save
        """
        self.output.write(frame)

        if not self.headless:
            cv2.imshow('output', frame)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                self.is_exiting = True

        self.frame_index += 1

        if self.frame_index % int(self.midgard.N / 10) == 0:
            print('{:.2f}'.format(self.frame_index / self.midgard.N * 100) + '%', self.frame_index, '/', self.midgard.N)

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
        if self.mode == Midgard.Mode.APPEARANCE_RGB:
            shutil.copy2(src, dst)
        else:
            img = cv2.imread(src)

            if img.shape != self.midgard.capture_shape:
                img = cv2.resize(img, self.midgard.capture_shape)

            if self.mode == Midgard.Mode.FLOW_UV:
                flow_uv = self.midgard.get_flow_uv(self.frame_index)
                flow_vis = get_flow_vis(flow_uv)
                cv2.imwrite(dst, flow_vis)
            elif self.mode == Midgard.Mode.FLOW_PROCESSED:
                self.midgard.get_frame()
                flow_uv = self.midgard.get_flow_uv(self.frame_index)
                self.detector.get_affine_matrix(flow_uv)
                self.detector.flow_vec_subtract(flow_uv)
                cv2.imwrite(dst, self.detector.flow_uv_warped_mag * 255 / np.max(self.detector.flow_uv_warped_mag))

    def process_annot(self, src: str, dst: str) -> None:
        """Processes an annotation file of the dataset and places it in the target directory

        Args:
            src (str): source annotation file path
            dst (str): destination annotation file path
        """
        with open(dst, 'w') as f:
            ann = self.midgard.get_midgard_annotation(self.frame_index, src)
            f.writelines(self.annotation_to_yolo(ann))

    def prepare_sequence(self, sequence: str) -> None:
        """Prepare a sequence of the MIDGARD dataset

        Args:
            sequence (str): which sequence to prepare, for example 'indoor-modern/sports-hall'
        """
        print(f'Preparing sequence {sequence}')
        self.sequence = sequence
        self.img_path = f'{self.midgard_path}/{self.sequence}/images'
        self.flo_path = f'{self.img_path}/output/inference/run.epoch-0-flow-field'
        self.ann_path = f'{self.midgard_path}/{self.sequence}/annotation'

        self.capture_shape = self.midgard.get_capture_shape()
        self.resolution = np.array(self.capture_shape)[:2][::-1]

        images = glob.glob(f'{self.img_path}/*.png')
        annotations = glob.glob(f'{self.ann_path}/*.csv')
        flow_fields = glob.glob(f'{self.flo_path}/*.flo')
        images.sort()
        annotations.sort()

        self.frame_index = 0
        self.N = len(images)

        if len(images) != len(annotations) or len(images) - 1 != len(flow_fields):
            print('Input counts: (images, annotations, flow fields):', len(images), len(annotations), len(flow_fields))
            raise ValueError('Input sizes do not match.')

        for i, (img_src, ann_src) in enumerate(zip(images, annotations)):
            # Skip the last frame for optical flow inputs, as it does not exist.
            if not (self.mode != Midgard.Mode.APPEARANCE_RGB and self.frame_index >= self.N - 2):
                self.process_image(img_src, f'{self.img_dest_path}/{self.output_index:06d}.png')
                self.process_annot(ann_src, f'{self.ann_dest_path}/{self.output_index:06d}.txt')
                self.output_index += 1
                self.frame_index += 1

    def process(self, mode: Midgard.Mode) -> None:
        """Processes the MIDGARD dataset"""
        channel_options = {
            Midgard.Mode.APPEARANCE_RGB: 3,
            Midgard.Mode.FLOW_UV: 2,
            Midgard.Mode.FLOW_UV_NORMALISED: 2,
            Midgard.Mode.FLOW_RADIAL: 1,
            Midgard.Mode.FLOW_PROCESSED: 1,
        }

        self.midgard_path = os.environ['MIDGARD_PATH']
        self.dest_path = os.environ['YOLOv4_PATH'] + '/dataset'
        self.img_dest_path = f'{self.dest_path}/images'
        self.ann_dest_path = f'{self.dest_path}/labels/yolo'
        sequences = [
            'countryside-natural/north-narrow',
            'countryside-natural/south-narrow',
            'indoor-historical/church',
            'indoor-historical/stairwell',
            'indoor-modern/glass-cube',
            'indoor-modern/sports-hall',
            # 'indoor-modern/warehouse-interior',
            'indoor-modern/warehouse-transition',
            'outdoor-historical/church',
            # 'semi-urban/island-north',
            'semi-urban/island-south',
            'urban/appartment-buildings',
        ]

        self.mode = mode
        self.channels = channel_options[self.mode]
        self.output_index = 0

        self.remove_contents_of_folder(self.img_dest_path)
        self.remove_contents_of_folder(self.ann_dest_path)

        print(f'Mode: {self.mode}')

        for sequence in sequences:
            self.prepare_sequence(sequence)



    def run(self):
        while self.is_active():
            orig_frame = self.midgard.get_frame()
            self.flow_uv = self.midgard.get_flow_uv(self.frame_index)
            self.flow_vis = get_flow_vis(self.flow_uv)
            self.midgard.get_midgard_annotation(self.frame_index)

            self.detector.get_affine_matrix(self.flow_uv)
            flow_uv_warped_vis, _ = self.detector.flow_vec_subtract(self.flow_uv)

            if self.debug_mode:
                flow_diff_vis, blocks_vis = self.detector.block_method(self.flow_uv)
                cluster_vis, _ = self.detector.clustering(self.detector.flow_uv_warped_mag)
                blocks_vis, _ = self.detector.clustering(self.detector.flow_diff_mag)
                summed_mag = self.detector.get_history(self.flow_uv)
                self.detector.draw(orig_frame)

                top_frames = np.hstack((orig_frame, flow_diff_vis, flow_uv_warped_vis, summed_mag))
                bottom_frames = np.hstack((self.flow_vis, blocks_vis, cluster_vis, summed_mag))
                self.write(np.vstack((top_frames, bottom_frames)))
            else:
                cluster_vis = self.detector.clustering(self.detector.flow_uv_warped_mag, True)
                self.write(cluster_vis)

    def release(self) -> None:
        """Release all media resources"""
        self.midgard.release()
        self.output.release()
