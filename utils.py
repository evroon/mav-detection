import cv2
import os
import numpy as np


def get_output(filename, capture, is_grey=False):
    path = 'media/output/{}.mp4'.format(filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    capture_size = (int(capture.get(3)), int(capture.get(4)))
    return cv2.VideoWriter(path, fourcc, 30.0, capture_size, not is_grey)

def get_sequence_length(path):
    return count_dir(path)

# Vis-Drone
def get_vis_drone_path(sequence):
    vis_drone_path = os.environ['VIS_DRONE_PATH']
    return vis_drone_path + '/sequences/{}'.format(sequence)

def get_vis_drone_capture(sequence):
    path = get_vis_drone_path(sequence)
    return cv2.VideoCapture(path + '/%7d.jpg'), count_dir(path)

# KITTI
def get_kitti_path(sequence):
    kitti_path = os.environ['KITTI_PATH']
    img_path = '{}/data_odometry_gray/dataset/sequences/{}/image_0'.format(kitti_path, sequence)
    # pose_path = '{}/data_odometry_poses/dataset/poses/00.txt'.format(kitti_path)
    return img_path

def get_kitti_capture(sequence):
    path = get_kitti_path(sequence)
    return cv2.VideoCapture(path + '/%6d.png'), count_dir(path)

def get_train_capture():
    path = 'media/train.mp4'
    return cv2.VideoCapture(path), 1e4

def count_dir(path):
    return len(os.listdir(path))


# Math utils
def line_intersection(line1, line2):
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

def line_angle(diff1, diff2):
    return np.arccos(np.dot(diff1, diff2) / (np.linalg.norm(diff1) * np.linalg.norm(diff2)))