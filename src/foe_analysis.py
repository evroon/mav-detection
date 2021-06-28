import json
from types import resolve_bases
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import json

import im_helpers

def get_error_img(foe_location: str = 'center') -> np.ndarray:
    resolution = (1920, 1024)
    img_shape = (resolution[1], resolution[0], 2)
    result_img = np.zeros((resolution[1], resolution[0]))

    base_path = os.getenv('SIMDATA_PATH')
    validation_data = glob.glob(f'{base_path}/mountains-demo/lake-foe_demo_{foe_location}-0-north-low-5.0-0-default/results/image_*.json')
    validation_data.sort()
    validation_data = validation_data[100:200]

    for i, json_path in enumerate(validation_data):
        if i % int(len(validation_data) / 10) == 0:
            print(f'{i / len(validation_data) * 100:.2f}% {i} / {len(validation_data)}')
        with open(json_path, 'r') as f:
            data = json.load(f)

            foe_gt = data['foe_gt']
            foe_dense = data['foe_dense']

            diff_gt = np.zeros(img_shape)
            diff_dense = np.zeros(img_shape)
            x_coords = np.tile(np.arange(resolution[0]), (resolution[1], 1))
            y_coords = np.tile(np.arange(resolution[1]), (resolution[0], 1)).T

            diff_gt[..., 0] = x_coords - foe_gt[0]
            diff_gt[..., 1] = y_coords - foe_gt[1]
            diff_dense[..., 0] = x_coords - foe_dense[0]
            diff_dense[..., 1] = y_coords - foe_dense[1]

            flow_magnitude = im_helpers.get_magnitude(diff_gt)
            img_distance = im_helpers.get_magnitude(diff_dense)
            norm = np.maximum(np.ones_like(flow_magnitude) * 1e-6, flow_magnitude * img_distance)

            arccos_arg = (diff_gt[..., 0] * diff_dense[..., 0] + diff_gt[..., 1] * diff_dense[..., 1]) / norm
            arccos_arg = np.clip(arccos_arg, -1, 1)
            angle_diff = np.rad2deg(np.arccos(arccos_arg))

            result_img += angle_diff

    return result_img

img_left = get_error_img('left')
img_center = get_error_img('center')
img_right = get_error_img('right')

total = img_left + img_center + img_right
img = im_helpers.apply_colormap(total)
cv2.imwrite('media/foe-error.png', img)
