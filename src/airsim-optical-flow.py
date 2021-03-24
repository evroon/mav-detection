import os
import glob
import json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import airsim
import cv2
from sklearn.preprocessing import normalize

def mat4x4_vec4_mult(mat4x4, vec4):
    return np.array(
        [
            mat4x4[0, 0] * vec4[..., 0] + mat4x4[0, 1] * vec4[..., 1] + mat4x4[0, 2] * vec4[..., 2] + mat4x4[0, 3] * vec4[..., 3],
            mat4x4[1, 0] * vec4[..., 0] + mat4x4[1, 1] * vec4[..., 1] + mat4x4[1, 2] * vec4[..., 2] + mat4x4[1, 3] * vec4[..., 3],
            mat4x4[2, 0] * vec4[..., 0] + mat4x4[2, 1] * vec4[..., 1] + mat4x4[2, 2] * vec4[..., 2] + mat4x4[2, 3] * vec4[..., 3],
            mat4x4[3, 0] * vec4[..., 0] + mat4x4[3, 1] * vec4[..., 1] + mat4x4[3, 2] * vec4[..., 2] + mat4x4[3, 3] * vec4[..., 3]
        ]
    ).T.swapaxes(0, 1)


def world_to_screen(view_proj, screen_res, world_pos):
    world_pos_4d = np.zeros((world_pos.shape[0], world_pos.shape[1], 4))
    world_pos_4d[..., :3] = world_pos
    world_pos_4d[..., 3] = 1.0
    pos_screen = mat4x4_vec4_mult(view_proj, world_pos_4d)

    rhw = 1.0 / pos_screen[..., 3]
    for i in range(3):
        pos_screen[..., i] *= rhw

    normalized_x = +pos_screen[..., 0] * 0.5 + 0.5
    normalized_y = -pos_screen[..., 1] * 0.5 + 0.5

    result = np.zeros((world_pos.shape[0], world_pos.shape[1], 2))
    result[..., 0] = normalized_x * screen_res[0]
    result[..., 1] = normalized_y * screen_res[1]

    return result

def screen_to_world(view_proj_inv, screen_res, screen_pos, depth):
    NormalizedX = screen_pos[..., 0] / screen_res[0]
    NormalizedY = screen_pos[..., 1] / screen_res[1]

    ScreenSpaceX = 2.0 * (NormalizedX - 0.5)
    ScreenSpaceY = 2.0 * ((1.0 - NormalizedY) - 0.5)

    RayStartProjectionSpace = np.zeros((depth.shape[0], depth.shape[1], 4))
    RayStartProjectionSpace[..., 0] = ScreenSpaceX
    RayStartProjectionSpace[..., 1] = ScreenSpaceY
    RayStartProjectionSpace[..., 2] = 1
    RayStartProjectionSpace[..., 3] = 1

    RayEndProjectionSpace = np.copy(RayStartProjectionSpace)
    RayEndProjectionSpace[..., 2] = 0.5

    HGRayStartWorldSpace = mat4x4_vec4_mult(view_proj_inv, RayStartProjectionSpace)
    HGRayEndWorldSpace = mat4x4_vec4_mult(view_proj_inv, RayEndProjectionSpace)

    RayStartWorldSpace = HGRayStartWorldSpace[..., :-1]
    RayEndWorldSpace = HGRayEndWorldSpace[..., :-1]

    for i in range(3):
        RayStartWorldSpace[..., i] /= HGRayStartWorldSpace[..., 3]
        RayEndWorldSpace[..., i] /= HGRayEndWorldSpace[..., 3]

    norm = np.linalg.norm(RayEndWorldSpace - RayStartWorldSpace, axis=2)
    RayDirWorldSpace = np.zeros_like(RayEndWorldSpace)
    result = np.zeros_like(RayEndWorldSpace)

    for i in range(3):
        RayDirWorldSpace[..., i] = (RayEndWorldSpace[..., i] - RayStartWorldSpace[..., i]) / norm
        result[..., i] = RayStartWorldSpace[..., i] + (RayDirWorldSpace[..., i] * depth)

    return result

def get_view_proj_mat(f):
    state_dict = json.load(f)
    view_proj_str = state_dict['Drone1']['viewProjectionMatrix']
    view_proj = view_proj_str.replace('[', '').replace(']', '').strip().split(' ')
    view_proj = [float(x) for x in view_proj]
    return np.array(view_proj).reshape((4, 4)).T

def get_flow() -> np.ndarray:
    dir = 'data/states'
    states = glob.glob(f'{dir}/*.json')
    states = [x for x in states if 'timestamp' not in x]
    states.sort()
    screen_res = (1920, 1080)

    x_coords = np.tile(np.arange(screen_res[0]), (screen_res[1], 1))
    y_coords = np.tile(np.arange(screen_res[1]), (screen_res[0], 1)).T
    coords = np.stack((x_coords, y_coords)).swapaxes(0, 2)

    for i, state in enumerate(states[::-1]):
        with open(state, 'r') as f:
            view_proj = get_view_proj_mat(f)

            depth_img = np.array(
                airsim.read_pfm('data/mountains-stationary/lake-north-low-2.5-10-default/depths/image_00002.png')[0]
            ).T

            result = np.zeros_like(depth_img)
            view_proj_inv = np.linalg.inv(view_proj)

            world_pos = screen_to_world(view_proj_inv, screen_res, coords, depth_img)
            screen_pos = world_to_screen(view_proj, screen_res, world_pos)

            # print(screen_pos.shape)
            screen_pos = np.maximum(screen_pos, 0)
            screen_pos[..., 0] = np.minimum(screen_pos[..., 0], screen_res[0] - 1)
            screen_pos[..., 1] = np.minimum(screen_pos[..., 1], screen_res[1] - 1)

            screen_pos = np.round(screen_pos).astype(np.uint16)

            result[screen_pos[..., 0], screen_pos[..., 1]] = 255
            cv2.imwrite('result.png', result.T)
            return


get_flow()
