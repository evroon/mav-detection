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

    def __init__(self, base_path: str, logger: logging.Logger, sequence: str, img_dir: str = '/images', seq_dir: str = '') -> None:
        self.base_path = base_path
        self.logger = logger
        self.sequence = sequence

        if self.sequence == '':
            self.sequence = self.get_default_sequence()

        img_format: str = 'image_%05d.png'
        self.seq_path = f'{base_path}{seq_dir}/{self.sequence}'
        self.img_path = f'{self.seq_path}{img_dir}'
        self.seg_path = f'{self.seq_path}/segmentations'
        self.depth_path = f'{self.seq_path}/depths'
        self.gt_of_path = f'{self.seq_path}/optical-flow'
        self.ann_path = f'{self.seq_path}/annotation'
        self.results_path = f'{self.seq_path}/results'
        self.img_pngs = f'{self.img_path}/{img_format}'
        self.vid_path = f'{self.seq_path}/recording.mp4'

        self.jpg_to_png()
        self.reorder_pngs(self.img_path)
        self.reorder_pngs(self.seg_path)
        self.reorder_pngs(self.depth_path)

        if not os.path.exists(self.vid_path):
            utils.img_to_video(self.img_pngs, self.vid_path)

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

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        if len(os.listdir(self.ann_path)) < 1:
            self.create_annotations()

        if not os.path.exists(self.gt_of_path):
            self.create_ground_truth_optical_flow()

        if self.capture_size != self.flow_size:
            self.logger.warning(f'original capture with size {self.capture_size} does not match flow, which has size {self.flow_size}')

        if self.N != utils.get_frame_count(self.orig_capture) - 1:
            self.logger.error(f'Input counts: (images, flow fields): {utils.get_frame_count(self.orig_capture)}, {self.N}')
            self.run_flownet2()

        print('Dataset loaded.')

    def run_flownet2(self) -> None:
        self.logger.info('Running FlowNet2...')
        flownet2 = os.environ['FLOWNET2']
        subprocess.call([f'{flownet2}/launch_docker.sh', '--run', '--dataset',  f'{self.img_path}'])

    def get_default_sequence(self) -> str:
        raise ValueError('Not implemented.')

    def create_ground_truth_optical_flow(self) -> None:
        pass

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

    def jpg_to_png(self) -> None:
        """Converts JPGs (if they exist) into the correct PNG format."""
        for img in os.listdir(self.img_path):
            if os.path.splitext(img)[1] == '.jpg':
                img_path = self.img_path + '/' + img
                frame = cv2.imread(img_path)
                index = int(img.replace('.jpg', ''))
                cv2.imwrite(f'{self.img_path}/{os.path.dirname(img)}/image_{index:05d}.png', frame)
                os.remove(img_path)

    def reorder_pngs(self, base_path: str) -> None:
        """Lets the image indices start at 0."""
        pngs = glob.glob(base_path + '/image_*')
        pngs.sort()

        for i, png in enumerate(pngs):
            extension = os.path.splitext(png)[-1]
            shutil.move(png, f'{base_path}/image_{i:05d}{extension}')

    def get_angular_velocity(self, i:int) -> np.ndarray:
        pass

    def get_delta_time(self, i:int) -> float:
        pass

    def get_gt_foe(self, i:int) -> Optional[Tuple[float, float]]:
        """Returns the ground truth Focus of Expansion.

        Args:
            i (int): frame index

        Returns:
            Optional[Tuple[float, float]]: Focus of Expansion
        """
        return None

    def release(self) -> None:
        """Release all media resources"""
        self.orig_capture.release()
        self.flow_capture.release()
        cv2.destroyAllWindows()
