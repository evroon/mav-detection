import utils
import cv2
import os
import numpy as np
import torch
import airsim
import shutil
import logging
import subprocess
from typing import Optional, Tuple, cast, List

import im_helpers


class Dataset:
    '''Desscribes a dataset with images, annotations and flow fields.'''

    def __init__(self, base_path: str, logger: logging.Logger, sequence: str, img_dir: str = '/images', seq_dir: str = '') -> None:
        self.base_path = base_path
        self.logger = logger
        self.sequence = sequence

        if self.sequence == '':
            self.sequence = self.get_default_sequence()

        img_format: str = 'image_%05d.png'
        self.seq_path = f'{base_path}{seq_dir}/{self.sequence}'
        self.img_path = f'{self.seq_path}{img_dir}'
        self.seg_path = f'{self.seq_path}/segmentations'
        self.depth_path = f'{self.seq_path}/depths'
        self.depth_pfms = f'{self.depth_path}/image_*.pfm'
        self.depth_vis_path = f'{self.seq_path}/depth-vis'
        self.gt_of_path = f'{self.seq_path}/optical-flow'
        self.gt_of_vis_path = f'{self.seq_path}/optical-flow-vis'
        self.ann_path = f'{self.seq_path}/annotation'
        self.results_path = f'{self.seq_path}/results'
        self.result_imgs_path = f'{self.seq_path}/result-images'
        self.img_pngs = f'{self.img_path}/{img_format}'
        self.img_pngs_ffmpeg = f'{self.img_path}/image_%5d.png'
        self.img_pngs_glob = f'{self.img_path}/image_*.png'
        self.vid_path = f'{self.seq_path}/recording.mp4'
        self.state_path = f'{self.seq_path}/states'
        self.half_res_img_path = f'{self.seq_path}/half-res-images'
        self.hrnet_out = f'{self.half_res_img_path}/hrnet'
        self.flownet_output = f'{self.img_path}/output/inference/run.epoch-0-flow-field'

        self.mp4_to_png()
        self.jpg_to_png()
        self.reorder_anns(self.ann_path)
        self.reorder_pngs(self.img_path)
        self.reorder_pngs(self.seg_path)
        self.reorder_pngs(self.depth_path)

        if not os.path.exists(self.vid_path):
            utils.img_to_video(self.img_pngs, self.vid_path)

        self.orig_capture = cv2.VideoCapture(self.img_pngs)

        if not os.path.exists(self.flownet_output):
            self.run_flownet2()

        self.flow_capture = cv2.VideoCapture(f'{self.img_path}/output/flownet2.mp4')
        self.capture_size = utils.get_capture_size(self.orig_capture)
        self.capture_shape = self.get_capture_shape()
        self.resolution: np.ndarray = np.array(self.capture_shape)[:2][::-1]
        self.flow_size = utils.get_capture_size(self.flow_capture)
        self.N = utils.get_frame_count(self.flow_capture)
        self.start_frame = 250

        if not os.path.exists(self.ann_path):
            os.makedirs(self.ann_path)

        if not os.path.exists(self.half_res_img_path):
            self.create_half_res_images()

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        if len(os.listdir(self.ann_path)) < 1:
            self.create_annotations()

        if not os.path.exists(self.gt_of_path) or not os.path.exists(self.gt_of_vis_path):
            self.create_ground_truth_optical_flow()

        if not os.path.exists(self.depth_vis_path):
            self.create_depth_visualisation()

        if self.capture_size != self.flow_size:
            self.logger.warning(f'original capture with size {self.capture_size} does not match flow, which has size {self.flow_size}')

        if not os.path.exists(self.hrnet_out):
            self.run_hrnet()

        self.logger.info('Dataset loaded.')

    def create_half_res_images(self) -> None:
        """Creates a directory with the images with half the original resolution."""

        utils.create_if_not_exists(self.half_res_img_path)
        pngs = utils.sorted_glob(self.img_pngs_glob)

        for orig_path in pngs:
            img = cv2.imread(orig_path)
            img_half = im_helpers.resize_percent(img, 50)
            cv2.imwrite(f'{self.half_res_img_path}/{os.path.basename(orig_path)}', img_half)

    def run_hrnet(self) -> None:
        """Runs HRNet-OCR on the current sequence."""
        if torch.cuda.device_count() < 1:
            raise SystemError('There are no active GPUs.')

        self.logger.info('Running HRNet-OCR...')
        hrnet = os.environ['HRNET_PATH']
        subprocess.call([f'{hrnet}/launch_docker.sh', '--run', '--dataset',  self.half_res_img_path])

    def run_flownet2(self) -> None:
        """Runs FlowNet2 on the current sequence."""
        if torch.cuda.device_count() < 1:
            raise SystemError('There are no active GPUs.')

        self.logger.info('Running FlowNet2...')
        flownet2 = os.environ['FLOWNET2']
        subprocess.call([f'{flownet2}/launch_docker.sh', '--run', '--dataset',  self.img_path])

    def get_default_sequence(self) -> str:
        """The default sequence to use if no sequence was specified by user

        Raises:
            ValueError: If the method is not called on an inherited class

        Returns:
            str: the name of the default sequence
        """
        raise ValueError('Not implemented.')

    def get_state_filenames(self) -> List[str]:
        """Get the filenames of the state data.

        Returns:
            List[str]: filenames of json files containing state data
        """
        return []

    def create_ground_truth_optical_flow(self) -> None:
        """Creates ground truth optical flow if possible."""
        pass

    def create_depth_visualisation(self) -> None:
        """Creates depth visualisation images if possible."""
        pass

    def get_sky_segmentation(self, i: int) -> np.ndarray:
        img = cv2.imread(f'{self.hrnet_out}/image_{i:05d}_prediction.png')
        img = cv2.resize(img, (1920, 1080))

        # Segment sky only
        mask = (img[..., 0] == 180) * (img[..., 1] == 130)
        return mask

    def get_segmentation(self, i: int) -> np.ndarray:
        """Returns the segmentation mask image.

        Args:
            i (int): the current frame index

        Returns:
            np.ndarray: the segmentation mask
        """
        return cv2.imread(f'{self.seg_path}/image_{i:05d}.png')

    def validate_sky_segment(self, sky_mask: np.ndarray, depth_buffer: np.ndarray) -> Tuple[float, float]:
        sky_mask_gt = depth_buffer > 0.80 * np.max(depth_buffer)
        return im_helpers.calculate_tpr_fpr(sky_mask_gt * 255, sky_mask)

    def create_annotations(self) -> None:
        """Creates annotations in YOLOv4 format if possible."""
        pass

    def get_annotation(self, i: int, ann_path: str = None) -> List[utils.Rectangle]:
        """Returns a list of ground truth bounding boxes given an annotation file.

        Args:
            ann_path (str): the annotation .txt file to process, current file if None

        Returns:
            list: list of rectangles describing the ground truth bounding boxes
        """
        result = []

        if ann_path is None:
            ann_path = f'{self.ann_path}/image_{i:05d}.txt'

        with open(ann_path, 'r') as f:
            for line in f.readlines():
                values = [float(x) for x in line.split(' ')]
                rect = utils.Rectangle.from_yolo_input(values, self.resolution)
                if rect.get_area() > 1:
                    result.append(rect)

        self.ground_truth = result
        return result

    def get_flow_uv(self, i: int) -> np.ndarray:
        """Get the content of the .flo file for the current frame

        Returns:
            np.ndarray: (w, h, 2) array with flow vectors
        """
        flo_path = f'{self.img_path}/output/inference/run.epoch-0-flow-field/{i:06d}.flo'
        flow_uv = utils.read_flow(flo_path)

        if self.capture_size != self.flow_size:
            flow_uv = cv2.resize(flow_uv, self.capture_size)
            flow_uv[..., 1] *= self.capture_size[1] / self.flow_size[1]

        return flow_uv

    def get_capture_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the original image inputs.

        Returns:
            tuple: image shape
        """
        img = np.array(cv2.imread(f'{self.img_path}/image_00000.png'))
        return cast(Tuple[int, int, int], img.shape)

    def get_frame(self) -> np.ndarray:
        """Loads the frames of the next iteration

        Returns:
            np.ndarray: the raw BGR input frame
        """
        _, orig_frame = self.orig_capture.read()
        return orig_frame

    def mp4_to_png(self) -> None:
        """Converts MP4 into the correct PNG format (if they do not already exist)."""
        utils.create_if_not_exists(self.img_path)

        if len(os.listdir(self.img_path)) < 1:
            print('Converting mp4 to pngs.')
            utils.video_to_img(self.vid_path, self.img_pngs_ffmpeg)

    def jpg_to_png(self) -> None:
        """Converts JPGs (if they exist) into the correct PNG format."""
        for img in os.listdir(self.img_path):
            if os.path.splitext(img)[1] == '.jpg':
                img_path = self.img_path + '/' + img
                frame = cv2.imread(img_path)
                index = int(img.replace('.jpg', ''))
                cv2.imwrite(f'{self.img_path}/{os.path.dirname(img)}/image_{index:05d}.png', frame)
                os.remove(img_path)

    def reorder_pngs(self, base_path: str) -> None:
        """Lets the image indices start at 0."""
        pngs = utils.sorted_glob(base_path + '/image_*')

        for i, png in enumerate(pngs):
            extension = os.path.splitext(png)[-1]
            shutil.move(png, f'{base_path}/image_{i:05d}{extension}')

    def reorder_anns(self, base_path: str) -> None:
        """Lets the annotation indices start at 0."""
        anns = utils.sorted_glob(base_path + '/image_*')

        for i, annotation in enumerate(anns):
            extension = os.path.splitext(annotation)[-1]
            shutil.move(annotation, f'{base_path}/image_{i:05d}{extension}')

    def get_orientation(self, i:int) -> np.ndarray:
        """Returns the orientation from the IMU if known.

        Args:
            i (int): Frame index

        Returns:
            np.ndarray: the euler angles in inertial frame (rad)
        """
        pass

    def get_angular_difference(self, first:int, second:int) -> np.ndarray:
        """Returns the difference in orientations of the IMU between two frames if known.

        Args:
            first (int): Frame index 1
            second (int): Frame index 2

        Returns:
            np.ndarray: the angular Euler rates (pitch, yaw, roll) in body frame (rad/dt)
        """
        pass

    def get_time(self, i:int) -> float:
        """Returns the time in seconds at the current frame

        Args:
            i (int): Current frame index

        Returns:
            float: Time in seconds
        """
        pass

    def get_delta_time(self, i:int) -> float:
        """Returns the time difference in seconds between the previous and current frame

        Args:
            i (int): Current frame index

        Returns:
            float: Time difference in seconds
        """
        pass

    def get_gt_foe(self, i:int) -> Optional[Tuple[float, float]]:
        """Returns the ground truth Focus of Expansion.

        Args:
            i (int): Frame index

        Returns:
            Optional[Tuple[float, float]]: Ground truth Focus of Expansion
        """
        return None

    def get_gt_of(self, i:int) -> Optional[np.ndarray]:
        """Returns the ground truth optical flow field for a given frame

        Args:
            i (int): Frame index

        Returns:
            Optional[np.ndarray]: the ground truth optical flow field
        """
        return None

    def get_depth(self, i:int) -> Optional[np.ndarray]:
        """Returns the depth buffer for a given frame

        Args:
            i (int): Frame index

        Returns:
            Optional[np.ndarray]: the ground truth depth buffer
        """

        img_path = f'{self.depth_path}/image_{i:05d}.pfm'
        return np.array(airsim.read_pfm(img_path)[0])

    def release(self) -> None:
        """Release all media resources"""
        self.orig_capture.release()
        self.flow_capture.release()
        cv2.destroyAllWindows()
