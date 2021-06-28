import requests
import cv2
import numpy as np
import os
import subprocess
import json
import warnings
from typing import Dict, Tuple, List, Optional, cast, Any
from matplotlib import pyplot as plt

import utils
import im_helpers
from run_config import RunConfig
from frame_result import FrameResult
from datasets.dataset import Dataset


class Validator:
    def __init__(self, config: RunConfig) -> None:
        self.host: str = 'http://192.168.178.235:8099'
        self.config = config
        self.ious = np.zeros(1)
        self.foe_error = np.zeros((1, 2))

    def start_yolo_inference(self) -> None:
        run: str = self.config.settings['yolo_train_weights'][str(self.config.mode)]
        subprocess.call(['./launch_docker.sh', '--inference-only',  '"{run}"'])

    def get_hash(self, filename: str) -> str:
        return subprocess.check_output(['sha1sum', filename]).decode("utf-8").split(' ')[0]

    def check_cache(self, hash: str, directory: str) -> Tuple[Optional[Dict[int, List[str]]], str]:
        """Checks whether an input_file has already a cached response.

        Args:
            hash (str):      the hash to check for cached results
            directory (str): directory of the cached responses

        Returns:
            Tuple[Optional[Dict[int, List[str]]], str]: the cached json file as dict if it exists and the json path
        """

        json_path = f'{directory}/{hash}.json'

        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                print(f'Using cached file {hash}')
                return json.load(f), json_path

        return None, json_path

    def get_config(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], requests.get(f'{self.host}/config').json())

    def get_run_timestamp(self) -> str:
        return str(self.get_config()['start_time'])

    def get_inference(self, input_file: str, output_file: str, use_default_weights: bool = False) -> Dict[int, List[str]]:
        boxes_dir = os.path.dirname(input_file) + f'/bounding-boxes'
        hash = self.get_hash(input_file) + '-' + self.get_run_timestamp()
        cache, json_path = self.check_cache(hash, boxes_dir)

        if cache is not None:
            return cache

        print(f'Requesting results for hash {hash}')

        headers = {'accept': 'application/json'}
        params = (
            ('use_default_weights', use_default_weights),
        )
        files = {
            'video': (input_file, open(input_file, 'rb')),
        }
        try:
            response = requests.post(
                f'{self.host}/predict_video', headers=headers, params=params, files=files)
            open(output_file, 'wb').write(response.content)

            response = requests.get(f'{self.host}/predict_video_boxes')
        except requests.exceptions.ConnectionError:
            print('Could not connect to host.')
            exit(-1)

        json_result = response.json()

        with open(json_path, 'w') as f:
            json.dump(json_result, f)

        return cast(Dict[int, List[str]], json_result)

    def parse_frames(self, frames: Dict[int, List[str]]) -> Dict[int, FrameResult]:
        """Parse a list of strings into a dict of bounding box rectangles.

        Args:
            frames (Dict[int, List[str]]): list of strings representing bounding boxes per frame

        Returns:
            Dict[int, List[Tuple[str, int, utils.Rectangle]]]: list of names, confidences and rectangles of bounding boxes per frame
        """
        result: Dict[int, FrameResult] = dict()
        for frame, boxes in frames.items():
            frame = int(frame)
            frameresult = FrameResult()
            result[frame] = frameresult

            for _, box in enumerate(boxes):
                box_split = box.split(' ')
                floats = [float(x) for x in box_split[1:]]
                name = box_split[0]
                confidence = int(box_split[1])
                rect = utils.Rectangle.from_yolo_output(floats[1:])
                # frameresult.add_box(name, confidence, rect)

        return result

    def run_validation(self, estimates: Optional[Dict[int, FrameResult]] = None) -> None:
        self.dataset = self.config.get_dataset()
        self.positives = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.frames: Dict[int, FrameResult] = dict()

        self.load_results()
        self.plot()
        self.plot_roc()

    def load_results(self) -> None:
        print('Loading detection results...')
        for i in range(self.dataset.N - 1):
            filename = f'{self.dataset.results_path}/image_{i:05d}.json'

            with open(filename, 'r') as f:
                json_result = json.load(f)

                self.frames[i] = FrameResult()
                self.frames[i].time = json_result['time']
                self.frames[i].tpr = json_result['tpr']
                self.frames[i].fpr = json_result['fpr']
                self.frames[i].tpr_fixed = json_result['tpr_fixed']
                self.frames[i].fpr_fixed = json_result['fpr_fixed']
                self.frames[i].sky_tpr = json_result['sky_tpr']
                self.frames[i].sky_fpr = json_result['sky_fpr']
                self.frames[i].foe_dense = json_result['foe_dense']
                self.frames[i].foe_gt = json_result['foe_gt']
                self.frames[i].drone_flow_pixels = json_result['drone_flow_pixels']
                self.frames[i].drone_size_pixels = json_result['drone_size_pixels']
                self.frames[i].center_phi = json_result['center_phi']

    def plot(self) -> None:
        utils.create_if_not_exists('media/output')
        plt.hist(self.ious, np.linspace(0.0, 1.0, 20))
        plt.grid()
        plt.xlabel('IoU')
        plt.ylabel('Frequency [frames]')
        plt.savefig('media/output/ious.png', bbox_inches='tight')
        plt.savefig('media/output/ious.eps', bbox_inches='tight')

        # Plot histogram of FoE errors.
        foe_dense = np.array([x[1].foe_dense for x in self.frames.items()])
        foe_gt = np.array([x[1].foe_gt for x in self.frames.items()])

        if foe_gt[0] is None:
            return

        # Give drone time to stabilize.
        self.foe_error = foe_dense[56:] - foe_gt[56:]

        outlier_threshold = 50.0
        inliers_list = []
        outliers_list = []

        for i in range(self.foe_error.shape[0]):
            if np.abs(self.foe_error[i, 0]) < outlier_threshold and np.abs(self.foe_error[i, 1]) < outlier_threshold:
                inliers_list.append(i)
            else:
                outliers_list.append(i)

        print(outliers_list)

        if len(inliers_list) > 0:
            inliers = np.array(inliers_list)
            foe_error_inliers = self.foe_error[inliers]
            mean_error = np.average(foe_error_inliers, axis=0)
            std_error = np.std(foe_error_inliers, axis=0)

            print(f'foe outliers: {self.foe_error.shape[0] - inliers.shape[0]}, average error:',
                f'({mean_error[0]:.2f}, {mean_error[1]:.2f}), std: ({std_error[0]:.1f}, {std_error[1]:.1f})')
        else:
            print('Error: no inliers in FoE estimates')

    def plot_roc(self) -> None:
        ''' Plots the ROC curve for different thresholds. '''

        # for threshold in self.thresholds:
        # Ignore warnings about nan values.
        warnings.filterwarnings('ignore')

        # Load data
        t = np.array([f.time for _, f in self.frames.items()])
        phi = np.array([float(f.center_phi) for _, f in self.frames.items()])
        x = np.array([f.fpr for _, f in self.frames.items()])

        tpr = np.array([f.tpr for _, f in self.frames.items()])
        tpr_fixed = np.array([f.tpr_fixed for _, f in self.frames.items()])
        fpr = np.array([f.fpr for _, f in self.frames.items()])
        fpr_fixed = np.array([f.fpr_fixed for _, f in self.frames.items()])

        flow_x = np.array([float(f.drone_flow_pixels[0]) for _, f in self.frames.items()])
        flow_y = np.array([float(f.drone_flow_pixels[1]) for _, f in self.frames.items()])
        size = np.array([int(f.drone_size_pixels) for _, f in self.frames.items()])

        flow_x = flow_x[~np.isnan(flow_x)]
        flow_y = flow_y[~np.isnan(flow_y)]

        # Phi vs TPR
        plt.figure()
        plt.grid()
        plt.plot(phi, tpr, ls='', marker='o')
        plt.xlabel(r'$\kappa$ [deg]')
        plt.ylabel('True Positive Rate')
        plt.ylim(0, 1.0)
        plt.xlim(-180, 0)

        print(f'size: {np.average(size):.3f}, {np.std(size):.1f}')
        print(f'flow x: {np.average(flow_x):.2f}, {np.std(flow_x):.1f}')
        print(f'flow y: {np.average(flow_y):.2f}, {np.std(flow_y):.1f}')

        plt.savefig(f'{self.dataset.seq_path}/tpr_vs_time_raw', bbox_inches='tight')

        # Cluster/window the data into bins to make plot more readble.
        bins_start = np.linspace(-180, 0, 40)
        # bins_end = np.linspace(-25, 0, 30)
        bins = np.concatenate([bins_start])

        def get_avg_std(phi: np.ndarray, pr: np.ndarray) -> np.ndarray:
            avg_std_tmp = np.zeros((len(bins), 3))
            pr_finite = pr[~np.isnan(pr)]

            for i in range(1, len(bins)):
                bin_mask = (phi >= bins[i - 1]) * (phi < bins[i])
                bin_mask_pr = bin_mask[~np.isnan(pr)]

                avg_std_tmp[i-1, :] = [
                    np.average(phi[bin_mask]),
                    np.average(pr_finite[bin_mask_pr]),
                    np.std(pr_finite[bin_mask_pr])
                ]

            return avg_std_tmp

        avg_std_tpr = get_avg_std(phi, tpr)
        avg_std_tpr_fixed = get_avg_std(phi, tpr_fixed)
        avg_std_fpr = get_avg_std(phi, fpr)
        avg_std_fpr_fixed = get_avg_std(phi, fpr_fixed)

        plt.figure()
        plt.grid()
        plt.xlabel(r'$\kappa$ [deg]')
        plt.ylabel('True Positive Rate')
        plt.ylim(0, 1.0)
        plt.errorbar(avg_std_tpr[:, 0], avg_std_tpr[:, 1], yerr=avg_std_tpr[:, 2],
            marker='o', markersize=6, capsize=3, barsabove=True, label='', zorder=1, color='indigo')

        plt.savefig(f'{self.dataset.seq_path}/tpr_vs_time', bbox_inches='tight')

        # Save data
        np.save(
            f'{self.dataset.seq_path}/validation.npy',
            np.array([
                np.average(tpr), np.std(tpr),
                np.average(size), np.std(size),
                np.median(flow_x), np.std(flow_x),
                np.average(flow_y), np.std(flow_y),
                avg_std_tpr, avg_std_tpr_fixed,
                avg_std_fpr, avg_std_fpr_fixed,
                x, tpr,
                self.foe_error,
            ])
        )

        # Plot sky segmentation RoC
        x = np.array([f.sky_fpr for _, f in self.frames.items()])
        y = np.array([f.sky_tpr for _, f in self.frames.items()])

        x = x[:len(x)//2]
        y = y[:len(y)//2]

        plt.figure()
        plt.grid()
        plt.plot(x, y, ls='', marker='o')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim(0, 1.0)
        plt.savefig(f'{self.dataset.seq_path}/sky_roc', bbox_inches='tight')
        return

        threshold = 0.5

        # Cluster/window the data into bins to make plot more readble.
        bins = np.zeros(22)
        bins[:4] = np.linspace(0, 0.01, 4)
        bins[4:] = np.linspace(0.02, 1.0, 18) #np.max(x[x <= 1.0])
        avg_std = np.zeros((len(bins), 4))

        for i in range(1, len(bins)):
            data = np.where(np.logical_and(x > bins[i - 1], x <= bins[i]))
            avg_std[i, :] = [
                np.average(x[data]), np.average(y[data]),
                np.std(y[data]), np.std(x[data])
            ]

        no_gates_count = np.sum(x == 0)
        label = 'threshold: {:2.2f}, {:3} images without detections'.format(threshold, no_gates_count)

        # Remove nan values.
        avg_std = avg_std[avg_std[:, 0] <= 1.0, :]

        # Plot errorbars only for the optimal threshold.
        if True:
            plt.errorbar(avg_std[:, 0], avg_std[:, 1], avg_std[:, 2], avg_std[:, 3],
                marker='o', markersize=6, capsize=3, barsabove=False, label=label, zorder=1, color='indigo')
        else:
            plt.plot(avg_std[:, 0], avg_std[:, 1], label=label, marker='o', zorder=0, markersize=4, ls='--')

        print('Number of images with no gates detected: {} for threshold: {}'.format(no_gates_count, threshold))

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()
        plt.legend()
        plt.savefig('roc', bbox_inches='tight')
