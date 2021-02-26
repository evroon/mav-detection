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
from run_config import RunConfig
from frame_result import FrameResult
from midgard import Midgard


class Validator:
    def __init__(self, config: RunConfig) -> None:
        self.host: str = 'http://192.168.178.235:8099'
        self.config = config

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
            exit()

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
                frameresult.add_box(name, confidence, rect)

        return result

    def annotate(self, img: np.ndarray, boxes: FrameResult, ground_truth: List[utils.Rectangle]) -> List[float]:
        # Plot ground truth.
        for gt in ground_truth:
            self.positives += 1
            img = cv2.rectangle(
                img,
                gt.get_topleft_int(),
                gt.get_bottomright_int(),
                (0, 0, 255),
                3
            )

        ious: List[float] = []
        threshold: float = 0.5
        true_positives_in_frame = 0

        # Plot estimates.
        for box in boxes:
            max_iou = 0.0
            rect = box[2]

            # Determine detection quality.
            if rect.get_area() < 10:
                continue

            for gt in ground_truth:
                iou = utils.Rectangle.calculate_iou(gt, rect)
                if iou > max_iou:
                    max_iou = iou

            ious.append(max_iou)

            if max_iou > threshold:
                self.true_positives += 1
                true_positives_in_frame += 1
            else:
                self.false_positives += 1

            img = cv2.rectangle(
                img,
                rect.get_topleft_int(),
                rect.get_bottomright_int(),
                (0, 128, 255),
                3
            )
            img = cv2.putText(
                img,
                f'{box[0]}: {box[1]:02f}%, {max_iou:.02f}',
                rect.get_topleft_int_offset(),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 128, 255),
                2
            )

        self.false_negatives += len(ground_truth) - true_positives_in_frame
        return ious

    def run_validation(self, estimates: Optional[Dict[int, FrameResult]] = None) -> None:
        midgard = Midgard(self.config.logger, self.config.sequence)
        output = utils.get_output('evaluation', midgard.orig_capture)
        self.positives = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

        try:
            img_input = midgard.img_path + '/image_%5d.png'
            video_path = midgard.seq_path + '/video.mp4'
            video_annotated_path = midgard.seq_path + '/video-annotated.mp4'
            self.img_to_video(img_input, video_path)

            if estimates is None:
                frames_raw = self.get_inference(video_path, video_annotated_path, False)
                frames = self.parse_frames(frames_raw)
            else:
                frames = utils.assert_type(estimates)

            ious: List[float] = []

            for i in range(midgard.N):
                frame = midgard.get_frame()
                ground_truth = midgard.get_midgard_annotation(i)
                if i in frames:
                    ious_frame = self.annotate(frame, frames[i], ground_truth)
                    for x in ious_frame:
                        ious.append(x)
                    output.write(frame)

            # Save histogram of IoU values.
            ious = np.array(ious)
            plt.hist(ious, np.linspace(0.0, 1.0, 20))
            plt.grid()
            plt.xlabel('IoU')
            plt.ylabel('Frequency (frames)')
            plt.savefig('media/output/ious.png', bbox_inches='tight')

            if self.true_positives > 0:
                self.config.logger.info(f'TP: {self.true_positives}, FP: {self.false_positives}, FN: {self.false_negatives}')
            else:
                self.config.logger.error(f'No detections. TP: {self.true_positives}, FP: {self.false_positives}, FN: {self.false_negatives}')
        finally:
            output.release()
            self.write_results()

    def write_results(self) -> None:
        self.negatives = 0
        self.true_negatives = 0

        self.results = {
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'true_positive_rate': self.true_positives / max(1.0, self.positives),
            'true_negative_rate': self.true_negatives / max(1.0, self.negatives),
            'false_positive_rate': self.true_positives / max(1.0, self.false_positives + self.true_negatives),
            'false_negative_rate': self.true_negatives / max(1.0, self.false_negatives + self.true_positives),
            'recall': self.true_positives / max(1.0, self.true_positives + self.false_negatives),
            'precision': self.true_positives / max(1.0, self.true_positives + self.false_positives),
        }
        self.config.logger.info(self.results)

        output_file = f'results/{self.config}.json'
        if not os.path.exists(output_file):
            os.makedirs(os.path.dirname(output_file))
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=4)

        with open('main.csv', 'a') as csv_file:
            csv_file.write(','.join([str(x) for x in self.config]))
            csv_file.write(','.join([f'{self.results[x]:.06f}' for x in self.results]))
            csv_file.write('\n')

        self.plot_roc()

    def img_to_video(self, input: str, output: str) -> None:
        if not os.path.exists(output):
            command = f'ffmpeg -r 30 -i {input} -c:v libx264 -vf fps=30 -pix_fmt yuv420p {output} -y'.split(
                ' ')
            subprocess.call(command)

    def plot_roc(self) -> None:
        ''' Plots the ROC curve for different thresholds. '''

        # for threshold in self.thresholds:
        # Ignore warnings about nan values.
        warnings.filterwarnings('ignore')

        # Load data
        x = self.results['true_positive_rate']
        y = self.results['false_positive_rate']
        return

        threshold = 0.5

        # Cluster/window the data into bins to make plot more readble.
        bins = np.zeros(22)
        bins[:4] = np.linspace(0, 0.01, 4)
        bins[4:] = np.linspace(0.02, np.max(x[x <= 1.0]), 18)
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
