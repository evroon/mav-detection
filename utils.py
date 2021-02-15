from __future__ import annotations
import cv2
import os
import numpy as np
from typing import Tuple, List


class Rectangle:
    def __init__(self, topleft, size):
        self.topleft = topleft
        self.size = size

    @classmethod
    def from_center(cls, center, size) -> Rectangle:
        return Rectangle(
            (center[0] - size[0] / 2, center[1] - size[1] / 2),
            size
        )

    @classmethod
    def from_points(cls, topleft, bottomright) -> Rectangle:
        return Rectangle(
            topleft,
            (bottomright[0] - topleft[0], bottomright[1] - topleft[1])
        )

    @classmethod
    def from_yolo(cls, arr: List[float]) -> Rectangle:
        return Rectangle(
            (arr[0], arr[1]),
            (arr[2], arr[3])
        )

    def get_topleft(self) -> Tuple[float, float]:
        return (self.topleft[0], self.topleft[1])

    def get_bottomright(self) -> Tuple[float, float]:
        return (self.topleft[0] + self.size[0], self.topleft[1] + self.size[1])

    def get_topleft_int(self) -> Tuple[int, int]:
        return (int(self.topleft[0]), int(self.topleft[1]))

    def get_topleft_int_offset(self) -> Tuple[int, int]:
        return (int(self.topleft[0]), int(self.topleft[1]) - 5)

    def get_bottomright_int(self) -> Tuple[int, int]:
        return (int(self.topleft[0] + self.size[0]), int(self.topleft[1] + self.size[1]))

    def get_center(self) -> Tuple[float, float]:
        return (self.topleft[0] + self.size[0] / 2, self.topleft[1] + self.size[1] / 2)

    def get_center_int(self) -> Tuple[int, int]:
        return (int(self.topleft[0] + self.size[0] / 2), int(self.topleft[1] + self.size[1] / 2))

    def get_left(self) -> float:
        return self.topleft[0]

    def get_right(self) -> float:
        return self.topleft[0] + self.size[0]

    def get_top(self) -> float:
        return self.topleft[1]

    def get_bottom(self) -> float:
        return self.topleft[1] + self.size[1]

    def get_area(self) -> float:
        return self.size[0] * self.size[1]

    @classmethod
    def calculate_iou(cls, r1: Rectangle, r2: Rectangle) -> float:
        left = max(r1.get_left(), r2.get_left())
        right = min(r1.get_right(), r2.get_right())
        bottom = min(r1.get_bottom(), r2.get_bottom())
        top = max(r1.get_top(), r2.get_top())
        aoo = (right - left) * (bottom - top)
        aou = r1.get_area() + r2.get_area() - aoo
        return aoo / aou


def get_capture_size(capture: cv.VideoCapture) -> Tuple[int, int]:
    return (int(capture.get(3)), int(capture.get(4)))

def get_output(filename, capture=None, capture_size=None, is_grey=False):
    path = 'media/output/{}.mp4'.format(filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if capture_size is None:
        capture_size = get_capture_size(capture)

    return cv2.VideoWriter(path, fourcc, 15.0, capture_size, not is_grey)

def get_sequence_length(path: str) -> int:
    return count_dir(path)

def get_frame_count(cap: cv.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def get_fps(cap: cv.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FPS))

# Vis-Drone
def get_vis_drone_path(sequence: str) -> str:
    vis_drone_path = os.environ['VIS_DRONE_PATH']
    return vis_drone_path + '/sequences/{}'.format(sequence)

def get_vis_drone_capture(sequence: str) -> cv2.VideoCapture:
    path = get_vis_drone_path(sequence)
    return cv2.VideoCapture(path + '/%7d.jpg'), count_dir(path)

# KITTI
def get_kitti_path(sequence: str) -> str:
    kitti_path = os.environ['KITTI_PATH']
    img_path = '{}/data_odometry_gray/dataset/sequences/{}/image_0'.format(kitti_path, sequence)
    # pose_path = '{}/data_odometry_poses/dataset/poses/00.txt'.format(kitti_path)
    return img_path

def get_kitti_capture(sequence: str) -> cv2.VideoCapture:
    path = get_kitti_path(sequence)
    return cv2.VideoCapture(path + '/%6d.png'), count_dir(path)

# Cenek Albl et al.
def get_cenek_path(sequence: str, camera: int) -> List[str, str]:
    cenek_path = os.environ['CENEK_PATH']
    img_path = f'{cenek_path}/{sequence}/{camera}.mp4'
    ann_path = f'{cenek_path}/{sequence}/detections/{camera}.txt'
    return img_path, ann_path

def get_cenek_capture(sequence: str, camera: int) -> List[cv2.VideoCapture, int]:
    cap = cv2.VideoCapture(get_cenek_path(sequence, camera)[0])
    return cap, get_frame_count(cap)

def get_cenek_annotation(sequence: str, camera: int) -> str:
    return get_cenek_path(sequence, camera)[1]

def get_train_capture() -> cv2.VideoCapture:
    path = 'media/train.mp4'
    return cv2.VideoCapture(path), 1e4

def count_dir(path: str) -> int:
    return len(os.listdir(path))

# Math utils
def line_intersection(line1, line2) -> List[float, float]:
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return False, False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def line_angle(diff1, diff2) -> float:
    return np.arccos(np.dot(diff1, diff2) / (np.linalg.norm(diff1) * np.linalg.norm(diff2)))

def read_flow(filename: str) -> np.ndarray:
    TAG_FLOAT = 202021.25

    with open(filename, 'rb') as f:
        flo_number = np.fromfile(f, np.float32, count=1)[0]
        assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number

        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)

        return np.resize(data, (int(h), int(w), 2))

def blockshaped(arr: np.ndarray, nrows: int, ncols: int) -> np.ndarray:
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.

    Source: https://stackoverflow.com/a/16858283
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
