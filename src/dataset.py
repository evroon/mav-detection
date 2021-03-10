import utils
import cv2
import os
import numpy as np
import flow_vis
import glob
import shutil
import logging
import subprocess
from typing import Optional, Tuple, cast, List

from im_helpers import get_flow_radial, get_flow_vis

class Dataset:
    '''Desscribes a dataset with images, annotations and flow fields.'''

    def __init__(self, base_path: str, logger: logging.Logger, sequence: str) -> None:
        self.base_path = base_path
        self.logger = logger
        self.sequence = sequence

        if self.sequence == '':
            self.sequence = self.get_default_sequence()

        self.seq_path = f'{base_path}/{self.sequence}'
        self.img_path = f'{self.seq_path}/images'
        self.seg_path = f'{self.seq_path}/segmentations'
        self.ann_path = f'{self.seq_path}/annotation'
        self.img_pngs = f'{self.img_path}/image_%05d.png'
        self.vid_path = f'{self.seq_path}/recording.mp4'

        self.orig_capture = cv2.VideoCapture(self.img_pngs)
        self.flow_capture = cv2.VideoCapture(f'{self.img_path}/output/flownet2.mp4')
        self.capture_size = utils.get_capture_size(self.orig_capture)
        self.capture_shape = self.get_capture_shape()
        self.resolution: np.ndarray = np.array(self.capture_shape)[:2][::-1]
        self.flow_size = utils.get_capture_size(self.flow_capture)
        self.N = utils.get_frame_count(self.flow_capture)
        self.start_frame = 100

        if not os.path.exists(self.ann_path):
            os.makedirs(self.ann_path)

        if len(os.listdir(self.ann_path)) < 1:
            self.create_annotations()

        if not os.path.exists(self.vid_path):
            utils.img_to_video(self.img_pngs, self.vid_path)

        if self.capture_size != self.flow_size:
            self.logger.warning(f'original capture with size {self.capture_size} does not match flow, which has size {self.flow_size}')

        if self.N != utils.get_frame_count(self.orig_capture) - 1:
            self.logger.error(f'Input counts: (images, flow fields): {utils.get_frame_count(self.orig_capture)}, {self.N}')
            self.run_flownet2()

    def run_flownet2(self) -> None:
        self.logger.info('Running FlowNet2...')
        flownet2 = os.environ['FLOWNET2']
        subprocess.call([f'{flownet2}/launch_docker.sh', '--run', '--dataset',  f'{self.img_path}'])

    def get_default_sequence(self) -> str:
        return ''

    def create_annotations(self) -> None:
        pass

    def get_annotation(self, i: int, ann_path: str = None) -> List[utils.Rectangle]:
        """Returns a list of ground truth bounding boxes given an annotation file.

        Args:
            ann_path (str): the annotation .txt file to process, current file if None

        Returns:
            list: list of rectangles describing the ground truth bounding boxes
        """
        result = []

        if ann_path is None:
            ann_path = f'{self.ann_path}/image_{i:05d}.txt'

        with open(ann_path, 'r') as f:
            for line in f.readlines():
                values = [float(x) for x in line.split(' ')]
                result.append(utils.Rectangle.from_yolo_input(values, self.resolution))

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
        _, orig_frame = self.orig_capture.read()
        return orig_frame


    def release(self) -> None:
        """Release all media resources"""
        self.orig_capture.release()
        self.flow_capture.release()
        cv2.destroyAllWindows()
