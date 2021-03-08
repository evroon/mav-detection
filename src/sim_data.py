import numpy as np
import logging
import glob
import cv2
import re
import os

import utils
from dataset import Dataset

class SimData(Dataset):
    '''Helper functions for the MIDGARD dataset.'''

    def __init__(self, logger: logging.Logger, sequence: str) -> None:
        simdata_path = os.environ['SIMDATA_PATH']
        super().__init__(simdata_path, logger, sequence)

    def write_yolo_annotation(self, image_path: str) -> None:
        filename = os.path.basename(image_path)
        matches = re.findall('^image_(.+)[.]png$', filename)
        index = matches[0]

        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        img_size = np.array([width, height])
        start_x, start_y = -1, -1
        end_x, end_y = -1, -1

        for y in range(height):
            if np.sum(img[y, :, :]) > 0.0:
                end_y = y

                if start_y == -1:
                    start_y = y

        for x in range(width):
            if np.sum(img[:, x, :]) > 0.0:
                end_x = x

                if start_x == -1:
                    start_x = x

        rect = utils.Rectangle.from_points((start_x, start_y), (end_x, end_y))

        with open(f'{self.ann_path}/image_{index}.txt', 'w') as f:
            f.write(rect.to_yolo(img_size))

    def create_annotations(self) -> None:
        for image_path in glob.glob(f'{self.img_path}/image_*.png'):
            self.write_yolo_annotation(image_path)

    def get_default_sequence(self) -> str:
        return 'test/test'
