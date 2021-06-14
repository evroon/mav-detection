import numpy as np
import logging
import cv2
import re
import os
import json
import airsim
from typing import Optional, Tuple, List, Any
from scipy.spatial.transform import Rotation

import utils
import im_helpers
from datasets.dataset import Dataset
from airsim_optical_flow import write_flow

class SimData(Dataset):
    '''Helper functions for the AirSim synthetic dataset.'''

    def __init__(self, logger: logging.Logger, sequence: str) -> None:
        simdata_path = os.environ['SIMDATA_PATH']
        self.start_time = 0.0
        super().__init__(simdata_path, logger, sequence)
        self.start_time = self.get_time(0)

    def write_yolo_annotation(self, image_path: str) -> None:
        filename = os.path.basename(image_path)
        matches = re.findall('^image_(.+)[.]png$', filename)
        index = matches[0]

        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        img_size = np.array([width, height])

        rect = im_helpers.get_simple_bounding_box(img)

        with open(f'{self.ann_path}/image_{index}.txt', 'w') as f:
            f.write(rect.to_yolo(img_size))

    def get_state_filenames(self) -> List[str]:
        return utils.sorted_glob(f'{self.state_path}/1*.json')

    def get_state(self, i:int) -> Any:
        with open(self.get_state_filenames()[i], 'r') as f:
            return json.load(f)

    def get_orientation(self, i:int) -> np.ndarray:
        orientatation = self.get_state(i)['Drone1']['imu']['orientation']
        euler: np.ndarray = Rotation.from_quat([
            orientatation['x_val'],
            orientatation['y_val'],
            orientatation['z_val'],
            orientatation['w_val']
        ]).as_euler('xyz', degrees=False)
        return euler

    def get_angular_difference(self, first:int, second:int) -> np.ndarray:
        omega: np.ndarray = self.get_orientation(second) - self.get_orientation(first)
        omega = omega[[1, 2, 0]]
        omega[2] = -omega[2]
        return omega

    def get_time(self, i:int) -> float:
        timestamp: float = self.get_state(i)['Drone1']['imu']['time_stamp']
        return timestamp / 1e9 - self.start_time

    def get_delta_time(self, i:int) -> float:
        time_stamp1 = self.get_time(i-1)
        time_stamp2 = self.get_time(i)
        return float(time_stamp2 - time_stamp1)

    def get_gt_foe(self, i:int) -> Optional[Tuple[float, float]]:
        FoE = self.get_state(i)['Drone1']['ue4']['FoE']
        return (FoE['X'] * self.capture_size[0], FoE['Y'] * self.capture_size[1])

    def get_gt_of(self, i:int) -> Optional[np.ndarray]:
        flow_uv = utils.read_flow(f'{self.gt_of_path}/image_{i:05d}.flo')

        if self.capture_size != self.flow_size:
            flow_uv = cv2.resize(flow_uv, self.capture_size)

        return flow_uv

    def create_ground_truth_optical_flow(self) -> None:
        utils.create_if_not_exists(self.gt_of_path)
        utils.create_if_not_exists(self.gt_of_vis_path)
        write_flow(self)

    def create_depth_visualisation(self) -> None:
        print('Writing depth visualisations...')
        os.makedirs(self.depth_vis_path)
        sky_distance_factor = 5

        for i, img_path in enumerate(utils.sorted_glob(f'{self.depth_path}/image_*.pfm')):
            pfm_array = np.array(airsim.read_pfm(img_path)[0])
            depth_img = (pfm_array / np.max(pfm_array) * 255) * sky_distance_factor
            depth_img_int = np.clip(0, 255, depth_img).astype(np.uint8)
            depth_img_int = im_helpers.apply_colormap(depth_img_int)
            cv2.imwrite(f'{self.depth_vis_path}/image_{i:05d}.png', depth_img_int)

    def create_annotations(self) -> None:
        print('Creating YOLOv4 annotations...')
        for image_path in utils.sorted_glob(f'{self.seg_path}/image_*.png'):
            self.write_yolo_annotation(image_path)

    def get_default_sequence(self) -> str:
        return 'citypark-stationary/soccerfield-north-low-2.5-10-default'
