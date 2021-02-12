import utils
import cv2
import os
import numpy as np
import flow_vis
from enum import Enum
import glob
import shutil
from im_helpers import get_flow_radial, get_flow_vis
from typing import Optional, Tuple, cast, List


class Midgard:
    '''Helper functions for the MIDGARD dataset.'''
    class Mode(Enum):
        APPEARANCE_RGB = 0,
        FLOW_UV = 1,
        FLOW_UV_NORMALISED = 2,
        FLOW_RADIAL = 3,
        FLOW_PROCESSED = 4

    def __init__(self, sequence: str) -> None:
        midgard_path = os.environ['MIDGARD_PATH']
        self.sequence = sequence
        self.seq_path = f'{midgard_path}/{sequence}'
        self.img_path = f'{self.seq_path}/images'
        self.ann_path = f'{self.seq_path}/annotation'
        self.orig_capture = cv2.VideoCapture(f'{self.img_path}/image_%5d.png')
        self.flow_capture = cv2.VideoCapture(f'{self.img_path}/output/flownet2.mp4')
        self.capture_size = utils.get_capture_size(self.orig_capture)
        self.capture_shape = self.get_capture_shape()
        self.resolution = np.array(self.capture_shape)[:2][::-1]
        self.flow_size = utils.get_capture_size(self.flow_capture)
        self.N = utils.get_frame_count(self.flow_capture)
        self.start_frame = 100

        if self.capture_size != self.flow_size:
            print(f'Note: original capture with size {self.capture_size} does not match flow, which has size {self.flow_size}')

        if self.N != utils.get_frame_count(self.orig_capture) - 1:
            print('Input counts: (images, flow fields):', utils.get_frame_count(self.orig_capture), self.N)
            raise ValueError('Input sizes do not match.')

    def get_midgard_annotation(self, i: int, ann_path: str = None) -> List[utils.Rectangle]:
        """Returns a list of ground truth bounding boxes given an annotation file.

        Args:
            ann_path (str): the annotation .txt file to process, current file if None

        Returns:
            list: list of rectangles describing the ground truth bounding boxes
        """
        result = []

        if ann_path is None:
            ann_path = f'{self.ann_path}/annot_{i:05d}.csv'

        with open(ann_path, 'r') as f:
            for line in f.readlines():
                values = [float(x) for x in line.split(',')]

                values = [round(float(x)) for x in values]
                topleft = (values[1], values[2])
                result.append(utils.Rectangle(topleft, (values[3], values[4])))

        self.ground_truth = result
        return result

    def get_flow_uv(self, i: int) -> np.ndarray:
        """Get the content of the .flo file for the current frame

        Returns:
            np.ndarray: (w, h, 2) array with flow vectors
        """
        flo_path = f'{self.img_path}/output/inference/run.epoch-0-flow-field/{i:06d}.flo'
        flow_uv = utils.read_flow(flo_path)

        if self.capture_size != self.flow_size:
            flow_uv = cv2.resize(flow_uv, self.capture_size)

        return flow_uv

    def get_capture_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the original image inputs.

        Returns:
            tuple: image shape
        """
        img = np.array(cv2.imread(f'{self.img_path}/image_00000.png'))
        return cast(Tuple[int, int, int], img.shape)

    def get_frame(self) -> np.ndarray:
        """Loads the frames of the next iteration

        Returns:
            np.ndarray: the raw BGR input frame
        """
        s1, orig_frame = self.orig_capture.read()
        return orig_frame

    def release(self) -> None:
        """Release all media resources"""
        self.orig_capture.release()
        self.flow_capture.release()
        cv2.destroyAllWindows()
