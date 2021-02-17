import requests
import utils
import cv2
import numpy as np
import utils
from midgard import Midgard
from typing import Dict, Tuple, List, Optional, cast, Any
import os
import subprocess
import hashlib
import json
from matplotlib import pyplot as plt


host: str = 'http://192.168.178.235:8099'


class Validator:
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
        return cast(Dict[str, Any], requests.get(f'{host}/config').json())

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
                f'{host}/predict_video', headers=headers, params=params, files=files)
            open(output_file, 'wb').write(response.content)

            response = requests.get(f'{host}/predict_video_boxes')
        except requests.exceptions.ConnectionError:
            print('Could not connect to host.')
            exit()

        json_result = response.json()

        with open(json_path, 'w') as f:
            json.dump(json_result, f)

        return cast(Dict[int, List[str]], json_result)

    def parse_frames(self, frames: Dict[int, List[str]]) -> Dict[int, List[Tuple[str, int, utils.Rectangle]]]:
        """Parse a list of strings into a dict of bounding box rectangles.

        Args:
            frames (Dict[int, List[str]]): list of strings representing bounding boxes per frame

        Returns:
            Dict[int, List[Tuple[str, int, utils.Rectangle]]]: list of names, confidences and rectangles of bounding boxes per frame
        """
        result: Dict[int, List[Tuple[str, int, utils.Rectangle]]] = dict()
        for frame, boxes in frames.items():
            frame = int(frame)
            result[frame] = []
            for _, box in enumerate(boxes):
                box_split = box.split(' ')
                floats = [float(x) for x in box_split[1:]]
                name = box_split[0]
                confidence = int(box_split[1])
                rect = utils.Rectangle.from_yolo(floats[1:])
                result[frame].append((name, confidence, rect))

        return result

    def annotate(self, img: np.ndarray, boxes: List[Tuple[str, int, utils.Rectangle]], ground_truth: List[utils.Rectangle]) -> List[float]:
        # Plot ground truth.
        for gt in ground_truth:
            self.total_detections += 1
            img = cv2.rectangle(
                img,
                gt.get_topleft_int(),
                gt.get_bottomright_int(),
                (0, 0, 255),
                3
            )

        ious: List[float] = []
        threshold: float = 0.5

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
            self.estimated_detections += 1

            if max_iou > threshold:
                self.successful_detections += 1

            img = cv2.rectangle(
                img,
                rect.get_topleft_int(),
                rect.get_bottomright_int(),
                (0, 128, 255),
                3
            )
            img = cv2.putText(
                img,
                f'{box[0]}: {box[1]:02d}%, {max_iou:.02f}',
                rect.get_topleft_int_offset(),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 128, 255),
                2
            )
        return ious

    def run_validation(self, sequence: str, estimates: dict = None) -> None:
        midgard = Midgard(sequence)
        output = utils.get_output('evaluation', midgard.orig_capture)
        try:
            img_input = midgard.img_path + '/image_%5d.png'
            video_path = midgard.seq_path + '/video.mp4'
            video_annotated_path = midgard.seq_path + '/video-annotated.mp4'
            self.img_to_video(img_input, video_path)

            frames = self.get_inference(video_path, video_annotated_path, False)
            frames = self.parse_frames(frames)

            self.successful_detections = 0
            self.estimated_detections = 0
            self.total_detections = 0
            ious: List[float] = []

            for i in range(midgard.N):
                frame = midgard.get_frame()
                ground_truth = midgard.get_midgard_annotation(i)
                if i in frames:
                    ious_frame = self.annotate(frame, frames[i], ground_truth)
                    [ious.append(x) for x in ious_frame]
                    output.write(frame)

            # Save histogram of IoU values.
            ious = np.array(ious)
            plt.hist(ious, np.linspace(0.0, 1.0, 20))
            plt.grid()
            plt.xlabel('IoU')
            plt.ylabel('Frequency (frames)')
            plt.savefig('media/output/ious.png', bbox_inches='tight')
            print(f'{self.successful_detections} / {self.estimated_detections} / {self.total_detections}, {self.successful_detections / self.total_detections * 100:.02f}%')
        finally:
            output.release()
            self.write_results()

    def write_results(self) -> None:
        results = {
            'successful_detections': self.successful_detections,
            'estimated_detections': self.estimated_detections,
            'total_detections': self.total_detections
        }
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)


    def img_to_video(self, input: str, output: str) -> None:
        if not os.path.exists(output):
            command = f'ffmpeg -r 30 -i {input} -c:v libx264 -vf fps=30 -pix_fmt yuv420p {output} -y'.split(
                ' ')
            subprocess.call(command)
