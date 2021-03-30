import numpy as np
import logging
import glob
import cv2
import re
import os
import json
from typing import Optional, Tuple

import utils
from dataset import Dataset
from airsim_optical_flow import get_flow

class SimData(Dataset):
    '''Helper functions for the AirSim synthetic dataset.'''

    def __init__(self, logger: logging.Logger, sequence: str) -> None:
        simdata_path = os.environ['SIMDATA_PATH']
        super().__init__(simdata_path, logger, sequence)
        self.state_path = f'{self.seq_path}/states'
        self.states = glob.glob(f'{self.state_path}/*.json')
        self.states.sort()

    def write_yolo_annotation(self, image_path: str) -> None:
        filename = os.path.basename(image_path)
        matches = re.findall('^image_(.+)[.]png$', filename)
        index = matches[0]

        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        img_size = np.array([width, height])
        start_x, start_y = -1, -1
        end_x, end_y = -1, -1
        threshold = 0.0

        for y in range(height):
            if np.average(img[y, :, :]) > threshold:
                end_y = y

                if start_y == -1:
                    start_y = y

        for x in range(width):
            if np.average(img[:, x, :]) > threshold:
                end_x = x

                if start_x == -1:
                    start_x = x

        rect = utils.Rectangle.from_points((start_x, start_y), (end_x, end_y))

        with open(f'{self.ann_path}/image_{index}.txt', 'w') as f:
            f.write(rect.to_yolo(img_size))

    def get_state(self, i:int) -> np.ndarray:
        with open(self.states[i], 'r') as f:
            return json.load(f)

    def get_gt_foe(self, i:int) -> Optional[Tuple[float, float]]:
        FoE = self.get_state(i)['ue4']['Drone1']['FoE']
        return (FoE['X'] * 800, FoE['Y'] * 600)

    def create_ground_truth_optical_flow(self) -> None:
        os.makedirs(self.gt_of_path)
        get_flow(self.seq_path)
        pass

    def create_annotations(self) -> None:
        print('Creating YOLOv4 annotations...')
        for image_path in glob.glob(f'{self.seg_path}/image_*.png'):
            self.write_yolo_annotation(image_path)

    def get_default_sequence(self) -> str:
        return 'mountains-moving/lake-north-medium-5.0-10-default'
