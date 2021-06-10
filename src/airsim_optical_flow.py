import json
from airsim.types import Vector3r
import numpy as np
import airsim
import cv2
from typing import List, Tuple, Dict, Any, cast

from datasets.dataset import Dataset
import utils
from im_helpers import get_flow_vis

def mat4x4_vec4_mult(mat4x4: np.ndarray, vec4: np.ndarray) -> np.ndarray:
    return np.array(
        [
            mat4x4[0, 0] * vec4[..., 0] + mat4x4[0, 1] * vec4[..., 1] + mat4x4[0, 2] * vec4[..., 2] + mat4x4[0, 3] * vec4[..., 3],
            mat4x4[1, 0] * vec4[..., 0] + mat4x4[1, 1] * vec4[..., 1] + mat4x4[1, 2] * vec4[..., 2] + mat4x4[1, 3] * vec4[..., 3],
            mat4x4[2, 0] * vec4[..., 0] + mat4x4[2, 1] * vec4[..., 1] + mat4x4[2, 2] * vec4[..., 2] + mat4x4[2, 3] * vec4[..., 3],
            mat4x4[3, 0] * vec4[..., 0] + mat4x4[3, 1] * vec4[..., 1] + mat4x4[3, 2] * vec4[..., 2] + mat4x4[3, 3] * vec4[..., 3]
        ]
    ).T.swapaxes(0, 1)


def world_to_screen(view_proj: np.ndarray, screen_res: Tuple[int, int], world_pos: np.ndarray) -> np.ndarray:
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

def screen_to_world(view_proj_inv: np.ndarray, screen_res: Tuple[int, int], screen_pos: np.ndarray, depth: np.ndarray) -> np.ndarray:
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

def get_state(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return cast(Dict[str, Any], json.load(f))

def get_view_proj_mat(state: dict) -> np.ndarray:
    view_proj_str = state['Drone1']['ue4']['viewProjectionMatrix']
    view_proj = view_proj_str.replace('[', '').replace(']', '').strip().split(' ')
    view_proj = [float(x) for x in view_proj]
    return np.array(view_proj).reshape((4, 4)).T

def calculate_flow(
        view_proj1: np.ndarray,
        view_proj2: np.ndarray,
        screen_res: np.ndarray,
        screen_pos2: np.ndarray,
        depth_img: np.ndarray,
        drone_displacement: Vector3r,
        segmentation: np.ndarray) -> np.ndarray:

    view_proj2_inv = np.linalg.inv(view_proj2)

    world_pos = screen_to_world(view_proj2_inv, screen_res, screen_pos2, depth_img)

    # Subtract the drone velocity for all pixels in the mask in 3D world (unit: meters) frame.
    drone_mask = segmentation > 0
    world_pos[drone_mask, 0] -= drone_displacement.x_val
    world_pos[drone_mask, 1] -= drone_displacement.y_val
    world_pos[drone_mask, 2] -= drone_displacement.z_val

    screen_pos1 = world_to_screen(view_proj1, screen_res, world_pos)
    return screen_pos1 - screen_pos2

def write_flow(dataset: Dataset) -> np.ndarray:
    print('Calculating ground truth optical flow...')
    screen_res = dataset.capture_size
    states = [x for x in dataset.get_state_filenames() if 'timestamp' not in x]

    x_coords = np.tile(np.arange(screen_res[0]), (screen_res[1], 1))
    y_coords = np.tile(np.arange(screen_res[1]), (screen_res[0], 1)).T
    coords = np.stack((x_coords, y_coords)).swapaxes(0, 2)

    for i, _ in enumerate(states[1:]):
        if i % int(len(states[1:]) / 10) == 0:
            dataset.logger.info(f'{i / len(states[1:]) * 100:.2f}%')

        state1 = get_state(states[i-1])
        state2 = get_state(states[i])

        view_proj1 = get_view_proj_mat(state1)
        view_proj2 = get_view_proj_mat(state2)

        delta_time = dataset.get_delta_time(i)
        drone_velocity = state1['Drone2']['ue4']['linearVelocity']
        drone_displacement = Vector3r(drone_velocity['X'], drone_velocity['Y'], drone_velocity['Z']) * delta_time * 100
        if np.isnan(drone_displacement.x_val):
            drone_displacement = Vector3r()

        img_path = f'{dataset.depth_path}/image_{i:05d}.pfm'
        depth_img = np.array(airsim.read_pfm(img_path)[0]).T * 100

        segmentation_img = cv2.imread(f'{dataset.seg_path}/image_{i:05d}.png', 0).T

        result = calculate_flow(view_proj1, view_proj2, screen_res, coords, depth_img, drone_displacement, segmentation_img)

        # Transpose image, transpose and mirror flow vectors.
        result = -result[..., [0, 1]].swapaxes(0, 1)

        utils.write_flow(
            f'{dataset.gt_of_path}/image_{i:05d}.flo',
            result
        )

        cv2.imwrite(f'{dataset.gt_of_vis_path}/image_{i:05d}.png', get_flow_vis(result))
