import utils
import cv2
import os
import numpy as np
import flow_vis
from enum import Enum
import glob
import shutil
from im_helpers import get_flow_radial, get_flow_vis
from typing import Optional


class Midgard:
    '''Converts MIDGARD dataset for use with YOLOv4.'''
    class Mode(Enum):
        APPEARANCE_RGB = 0,
        FLOW_UV = 1,
        FLOW_UV_NORMALISED = 2,
        FLOW_RADIAL = 3,
        FLOW_PROCESSED = 4

    def __init__(self, sequence: str, debug_mode: bool) -> None:
        self.sequence = sequence
        self.debug_mode = debug_mode

        midgard_path = os.environ['MIDGARD_PATH']
        self.img_path = f'{midgard_path}/{sequence}/images'
        self.ann_path = f'{midgard_path}/{sequence}/annotation'
        self.orig_capture = cv2.VideoCapture(f'{self.img_path}/image_%5d.png')
        self.flow_capture = cv2.VideoCapture(f'{self.img_path}/output/color_coding/video/000000.flo.mp4')
        self.capture_size = utils.get_capture_size(self.orig_capture)
        self.capture_shape = self.get_capture_shape()
        self.resolution = np.array(self.capture_shape)[:2][::-1]
        self.flow_size = utils.get_capture_size(self.flow_capture)
        self.N = utils.get_frame_count(self.flow_capture)
        self.frame_columns, self.frame_rows = 1, 1

        if debug_mode:
            self.frame_columns, self.frame_rows = 4, 2

        capture_size = (self.capture_size[0] * self.frame_columns, self.capture_size[1] * self.frame_rows)
        self.output = utils.get_output('detection', capture_size=capture_size, is_grey=not debug_mode)
        self.input_index = 0
        self.start_frame = 100
        self.is_exiting = False

        if self.capture_size != self.flow_size:
            print(f'Note: original capture with size {self.capture_size} does not match flow, which has size {self.flow_size}')

        if self.N != utils.get_frame_count(self.orig_capture) - 1:
            print('Input counts: (images, flow fields):', utils.get_frame_count(self.orig_capture), self.N)
            raise ValueError('Input sizes do not match.')

    def annotation_to_yolo(self, rects: list) -> str:
        """Converts the rectangles to the text format read by YOLOv4

        Args:
            rects (list): the input rectangles

        Returns:
            str: a list of bounding boxes in format '0 {center_x} {center_y} {size_x} {size_y}' with coordinates in range [0, 1]
        """
        result: str = ''
        for rect in rects:
            center = np.array(rect.get_center()) / self.resolution.astype(np.float)
            size = np.array(rect.size) / (2.0 * self.resolution.astype(np.float))
            result += f'0 {center[0]} {center[1]} {size[0]} {size[1]}\n'

        return result

    def get_midgard_annotation(self, ann_path: str = None) -> list:
        """Returns a list of ground truth bounding boxes given an annotation file.

        Args:
            ann_path (str): the annotation .txt file to process, current file if None

        Returns:
            list: list of rectangles describing the ground truth bounding boxes
        """
        result = []

        if ann_path is None:
            ann_path = f'{self.ann_path}/annot_{self.input_index:05d}.csv'

        with open(ann_path, 'r') as f:
            for line in f.readlines():
                values = [float(x) for x in line.split(',')]

                values = [round(float(x)) for x in values]
                topleft = (values[1], values[2])
                result.append(utils.Rectangle(topleft, (values[3], values[4])))

        self.ground_truth = result
        return result

    def get_flow_uv(self) -> np.ndarray:
        """Get the content of the .flo file for the current frame

        Returns:
            np.ndarray: (w, h, 2) array with flow vectors
        """
        flo_path = f'{self.img_path}/output/inference/run.epoch-0-flow-field/{self.input_index:06d}.flo'
        flow_uv = utils.read_flow(flo_path)

        if self.capture_size != self.flow_size:
            flow_uv = cv2.resize(flow_uv, self.capture_size)

        return flow_uv

    def get_capture_shape(self) -> tuple:
        """Get the shape of the original image inputs.

        Returns:
            tuple: image shape
        """
        return np.array(cv2.imread(f'{self.img_path}/image_00000.png')).shape

    def get_frame(self) -> np.ndarray:
        """Loads the frames of the next iteration

        Returns:
            np.ndarray: the raw BGR input frame
        """
        s1, orig_frame = self.orig_capture.read()

        while self.debug_mode and self.input_index < self.start_frame:
            s1, orig_frame = self.orig_capture.read()
            self.input_index += 1

        self.orig_frame = orig_frame
        self.flow_uv = self.get_flow_uv()
        self.flow_vis = get_flow_vis(self.flow_uv)
        get_flow_radial(self.flow_vis)
        self.get_midgard_annotation()

        return orig_frame

    def is_active(self) -> bool:
        return self.input_index < self.N - 1 and not self.is_exiting

    def write(self, frame: np.ndarray) -> None:
        self.output.write(frame)

        if self.debug_mode:
            cv2.imshow('output', frame)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                self.is_exiting = True

        self.input_index += 1

        if self.input_index % int(self.N / 10) == 0:
            print('{:.2f}'.format(self.input_index / self.N * 100) + '%', self.input_index, '/', self.N)

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

            if img.shape != self.capture_shape:
                img = cv2.resize(img, self.capture_shape)

            if self.mode == Midgard.Mode.FLOW_UV:
                flow_uv = self.get_flow_uv()
                flow_vis = get_flow_vis(flow_uv)
                cv2.imwrite(dst, flow_vis)
            elif self.mode == Midgard.Mode.FLOW_PROCESSED:
                self.get_frame()
                self.detector.get_affine_matrix()
                self.detector.flow_vec_subtract()
                # cluster_vis = self.detector.clustering(self.detector.flow_uv_warped_mag, True)
                cv2.imwrite(dst, self.detector.flow_uv_warped_mag * 255 / np.max(self.detector.flow_uv_warped_mag))

    def process_annot(self, src: str, dst: str) -> None:
        """Processes an annotation file of the dataset and places it in the target directory

        Args:
            src (str): source annotation file path
            dst (str): destination annotation file path
        """
        with open(dst, 'w') as f:
            ann = self.get_midgard_annotation(src)
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

        self.capture_shape = self.get_capture_shape()
        self.resolution = np.array(self.capture_shape)[:2][::-1]

        images = glob.glob(f'{self.img_path}/*.png')
        annotations = glob.glob(f'{self.ann_path}/*.csv')
        flow_fields = glob.glob(f'{self.flo_path}/*.flo')
        images.sort()
        annotations.sort()

        self.input_index = 0
        self.N = len(images)

        if len(images) != len(annotations) or len(images) - 1 != len(flow_fields):
            print('Input counts: (images, annotations, flow fields):', len(images), len(annotations), len(flow_fields))
            raise ValueError('Input sizes do not match.')

        for i, (img_src, ann_src) in enumerate(zip(images, annotations)):
            # Skip the last frame for optical flow inputs, as it does not exist.
            if not (self.mode != Midgard.Mode.APPEARANCE_RGB and self.input_index >= self.N - 2):
                self.process_image(img_src, f'{self.img_dest_path}/{self.output_index:06d}.png')
                self.process_annot(ann_src, f'{self.ann_dest_path}/{self.output_index:06d}.txt')
                self.output_index += 1
                self.input_index += 1

    def process(self) -> None:
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

        self.mode = Midgard.Mode.FLOW_PROCESSED
        self.channels = channel_options[self.mode]
        self.output_index = 0

        self.remove_contents_of_folder(self.img_dest_path)
        self.remove_contents_of_folder(self.ann_dest_path)

        print(f'Mode: {self.mode}')

        for sequence in sequences:
            self.prepare_sequence(sequence)

    def release(self) -> None:
        """Release all media resources"""
        self.orig_capture.release()
        self.flow_capture.release()
        self.output.release()
        cv2.destroyAllWindows()
