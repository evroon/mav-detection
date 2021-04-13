from __future__ import annotations
import cv2
import os
import numpy as np
import subprocess
import json
import glob

from datetime import datetime
from typing import Tuple, List, Optional, TypeVar, cast, Dict, Any

from im_helpers import get_flow_vis


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

    return cv2.VideoWriter(filename, fourcc, 30.0, capture_size, not is_grey)


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
    """Read flow from .flo file

    Args:
        filename (str): path to the .flo file

    Returns:
        np.ndarray: resulting flow field
    """
    TAG_FLOAT = 202021.25

    with open(filename, 'rb') as f:
        flo_number = np.fromfile(f, np.float32, count=1)[0]
        assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number

        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)

        return np.resize(data, (int(h), int(w), 2))


def write_flow(filename: str, uv: np.ndarray, v: np.ndarray = None) -> None:
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """

    TAG_CHAR = np.array([202021.25], np.float32)
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height, width = u.shape

    with open(filename, 'wb') as f:
        # write the header
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width*nBands))
        tmp[:, np.arange(width)*2] = u
        tmp[:, np.arange(width)*2 + 1] = v
        tmp.astype(np.float32).tofile(f)


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
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def img_to_video(input: str, output: str, framerate: int = 30) -> None:
    """Convert images to a video

    Args:
        input (str): input images path
        output (str): output video path
        framerate (int, optional): the frame rate of the video. Defaults to 30.
    """
    if not os.path.exists(output):
        images = os.listdir(os.path.dirname(input))
        images = [f for f in images if 'image_' in os.path.basename(f)]
        images.sort()
        start_number = images[0].replace('image_', '').replace('.png', '')
        command = f'ffmpeg -start_number {start_number} -r {framerate} -i {input} -c:v libx264 -vf fps={framerate} -pix_fmt yuv420p {output} -y'
        subprocess.call(command.split(' '))


def is_rotation_matrix(R: np.ndarray) -> bool:
    """Checks if a matrix is a valid rotation matrix.

    Args:
        R (np.ndarray): the matrix to check

    Returns:
        bool: Whether the matrix is a rotation matrix
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return cast(bool, n < 1e-6)


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).

    Args:
        R (np.ndarray): the rotation matrix to process

    Returns:
        np.ndarray: the euler angles in degrees
    """
    assert(is_rotation_matrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.rad2deg(np.array([x, y, z]))


def get_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Convert object to JSON dictionary.

    Args:
        obj (Dict[str, Any]): object to convert

    Returns:
        Dict[str, Any]: resulting dictionary that can be saved as JSON
    """
    return cast(Dict[str, Any], json.loads(
        json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)))
    ))


def create_if_not_exists(dir: str) -> None:
    """Creates a directory if it does not already exist."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_magnitude(vector: np.ndarray) -> float:
    """Return the magnitude of a vector."""
    return float(np.sqrt(vector.x_val ** 2.0 + vector.y_val ** 2.0 + vector.z_val ** 2.0))


def get_time() -> datetime:
    return datetime.now()

def sorted_glob(path: str) -> List[str]:
    """Returns a sorted list of glob paths

    Args:
        path (str): the input path with a glob pattern

    Returns:
        List[str]: the paths matching the glob in alphabetical order
    """
    result = glob.glob(path)
    result.sort()
    return result
