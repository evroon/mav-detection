import os
import glob
import json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import airsim
import cv2
from sklearn.preprocessing import normalize


def world_to_screen(view_proj, screen_res, world_pos):
    world_pos_4d = np.array([*world_pos, 1.0])
    result = view_proj.dot(world_pos_4d)

    if True:
        rhw = 1.0 / result[3]
        pos_screen = np.array([result[0] * rhw, result[1] * rhw, result[2] * rhw, result[3]])

        normalized_x = pos_screen[0] * 0.5 + 0.5
        normalized_y = 1.0 - pos_screen[1] * 0.5 - 0.5

        return (
            normalized_x * screen_res[0],
            normalized_y * screen_res[1]
        )

def screen_to_world(view_proj_inv, screen_res, screen_pos, depth):
    NormalizedX = (screen_pos[0] - 0) / screen_res[0]
    NormalizedY = (screen_pos[1] - 0) / screen_res[1]

    ScreenSpaceX = 2.0 * (NormalizedX - 0.5)
    ScreenSpaceY = 2.0 * ((1.0 - NormalizedY) - 0.5)

    RayStartProjectionSpace = np.array([ScreenSpaceX, ScreenSpaceY, 1, 1])
    RayEndProjectionSpace = np.array([ScreenSpaceX, ScreenSpaceY, 0.5, 1])

    HGRayStartWorldSpace = view_proj_inv.dot(RayStartProjectionSpace)
    HGRayEndWorldSpace = view_proj_inv.dot(RayEndProjectionSpace)

    RayStartWorldSpace = HGRayStartWorldSpace[:-1]
    RayEndWorldSpace = HGRayEndWorldSpace[:-1]

    if HGRayStartWorldSpace[3] != 0.0:
        RayStartWorldSpace /= HGRayStartWorldSpace[3]

    if HGRayEndWorldSpace[3] != 0.0:
        RayEndWorldSpace /= HGRayEndWorldSpace[3]

    RayDirWorldSpace = (RayEndWorldSpace - RayStartWorldSpace) / np.linalg.norm(RayEndWorldSpace - RayStartWorldSpace)
    return RayStartWorldSpace + RayDirWorldSpace * depth

def get_view_proj_mat(f):
    state_dict = json.load(f)
    view_proj_str = state_dict['Drone1']['viewProjectionMatrix']
    view_proj = view_proj_str.replace('[', '').replace(']', '').split(' ')[:-1]
    view_proj = [float(x) for x in view_proj]
    return np.array(view_proj).reshape((4, 4)).T


def get_flow() -> np.ndarray:
    dir = 'data/states'
    states = glob.glob(f'{dir}/*.json')
    states = [x for x in states if 'timestamp' not in x]
    states.sort()

    for i, state in enumerate(states[::-1]):
        with open(state, 'r') as f:
            view_proj = get_view_proj_mat(f)

            depth_img = np.array(
                airsim.read_pfm('data/mountains-stationary/lake-north-low-2.5-10-default/depths/image_00002.png')[0]
            )
            result = np.zeros_like(depth_img)
            screen_res = (1280, 720)

            # test_world_pos = (31920, -6150, 910)
            # test_screen_pos = world_to_screen(view_proj, test_world_pos)
            # test_world_pos_reprojected = screen_to_world(view_proj, test_screen_pos, 32830)
            # print(test_screen_pos)
            # print(test_world_pos_reprojected)
            # print(depth_img[int(test_screen_pos[1] * 1080 / 720), int(test_screen_pos[0] * 1920 / 1280)])
            # print(img_3d[int(test_screen_pos[1]), int(test_screen_pos[0])])

            view_proj_inv = np.linalg.inv(view_proj)

            for y in range(depth_img.shape[0]):
                for x in range(depth_img.shape[1]):
                    world_pos = screen_to_world(view_proj_inv, screen_res, (x, y), depth_img[y, x])
                    screen_pos = world_to_screen(view_proj, screen_res, world_pos)
                    result[int(screen_pos[1]), int(screen_pos[0])] = 255

            cv2.imwrite('result.png', result)
            return


get_flow()
