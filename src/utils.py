from __future__ import annotations
import cv2
import os
import numpy as np
import subprocess
from typing import Tuple, List, Optional, TypeVar


class Rectangle:
    def __init__(self, topleft: Tuple[float, float], size: Tuple[float, float]) -> None:
        self.topleft: Tuple[float, float] = topleft
        self.size: Tuple[float, float] = size

    @classmethod
    def from_center(cls, center: Tuple[float, float], size: Tuple[float, float]) -> Rectangle:
        return Rectangle(
            (center[0] - size[0] / 2, center[1] - size[1] / 2),
            size
        )

    @classmethod
    def from_points(cls, topleft: Tuple[float, float], bottomright: Tuple[float, float]) -> Rectangle:
        return Rectangle(
            topleft,
            (bottomright[0] - topleft[0], bottomright[1] - topleft[1])
        )

    @classmethod
    def from_yolo_input(cls, arr: List[float], img_size: np.ndarray) -> Rectangle:
        center = np.array([arr[1], arr[2]]) * img_size.astype(np.float)
        size = np.array([arr[3], arr[4]]) * img_size.astype(np.float)
        return Rectangle.from_center(center, size)

    @classmethod
    def from_yolo_output(cls, arr: List[float]) -> Rectangle:
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

    def to_yolo(self, img_size: np.ndarray, obj_id: int = 0) -> str:
        img_size = img_size.astype(np.float)
        center = np.array(self.get_center()) / img_size
        size = np.array(self.size) / img_size
        return f'{obj_id} {center[0]} {center[1]} {size[0]} {size[1]}\n'

    @classmethod
    def calculate_iou(cls, r1: Rectangle, r2: Rectangle) -> float:
        """Calculates the Intersection over Union between two boxes.

        Args:
            r1 (Rectangle): box 1
            r2 (Rectangle): box 2

        Returns:
            float: the area of the overlap of the two boxes divided by the area of union.
        """
        left = max(r1.get_left(), r2.get_left())
        right = min(r1.get_right(), r2.get_right())
        bottom = min(r1.get_bottom(), r2.get_bottom())
        top = max(r1.get_top(), r2.get_top())
        aoo = (right - left) * (bottom - top)
        aou = r1.get_area() + r2.get_area() - aoo
        return aoo / aou

# Optional type cast helper
T = TypeVar('T')
def assert_type(arg: Optional[T]) -> T:
    assert arg is not None
    return arg


def get_capture_size(capture: cv2.VideoCapture) -> Tuple[int, int]:
    return (int(capture.get(3)), int(capture.get(4)))

def get_output(filename: str, capture: cv2.VideoCapture = None, capture_size: Tuple[int, int] = None, is_grey: bool = False) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if capture_size is None:
        capture_size = get_capture_size(capture)

    return cv2.VideoWriter(filename, fourcc, 15.0, capture_size, not is_grey)

def get_sequence_length(path: str) -> int:
    return count_dir(path)

def get_frame_count(cap: cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def get_fps(cap: cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FPS))


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
def get_cenek_path(sequence: str, camera: int) -> Tuple[str, str]:
    cenek_path = os.environ['CENEK_PATH']
    img_path = f'{cenek_path}/{sequence}/{camera}.mp4'
    ann_path = f'{cenek_path}/{sequence}/detections/{camera}.txt'
    return img_path, ann_path

def get_cenek_capture(sequence: str, camera: int) -> Tuple[cv2.VideoCapture, int]:
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
def line_intersection(line1: Tuple[Tuple[float, float], Tuple[float, float]], line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return False, False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def line_angle(diff1: float, diff2: float) -> float:
    return float(np.arccos(np.dot(diff1, diff2) / (np.linalg.norm(diff1) * np.linalg.norm(diff2))))

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

def img_to_video(input: str, output: str, framerate: int = 10) -> None:
    if not os.path.exists(output):
        images = os.listdir(os.path.dirname(input))
        images = [f for f in images if 'image_' in os.path.basename(f)]
        images.sort()
        start_number = images[0].replace('image_', '').replace('.png', '')
        command = f'ffmpeg -start_number {start_number} -r {framerate} -i {input} -c:v libx264 -vf fps={framerate} -pix_fmt yuv420p {output} -y'
        subprocess.call(command.split(' '))
