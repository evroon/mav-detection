import requests
import utils
import cv2
import numpy as np
import utils
from midgard import Midgard
from typing import Dict, Tuple, List, Optional, cast
import os
import subprocess
import hashlib
import json


host: str = 'http://192.168.178.235:8099'


class Validator:
    def get_hash(self, filename: str) -> str:
        return subprocess.check_output(['sha1sum', filename]).decode("utf-8").split(' ')[0]

    def check_cache(self, input_file: str, directory: str) -> Tuple[Optional[Dict[int, List[str]]], str]:
        """Checks whether an input_file has already a cached response.

        Args:
            input_file (str): the file to check for cached results
            directory (str): directory of the cached responses

        Returns:
            Tuple[Optional[Dict[int, List[str]]], str]: the cached json file as dict if it exists and the json path
        """
        hash = self.get_hash(input_file)
        json_path = f'{directory}/{hash}.json'

        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f), json_path

        return None, json_path

    def get_inference(self, input_file: str, output_file: str, use_default_weights: bool = False) -> Dict[int, List[str]]:
        boxes_dir = os.path.dirname(input_file) + f'/bounding-boxes'
        cache, json_path = self.check_cache(input_file, boxes_dir)

        if cache is not None:
            return cache

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
        result: Dict[int, List[Tuple[str, int, utils.Rectangle]]] = dict()
        for frame, boxes in frames.items():
            frame = int(frame)
            result[frame] = []
            for _, box in enumerate(boxes):
                box_split = box.split(' ')
                box_split[1:] = [float(x) for x in box_split[1:]]
                name = box_split[0]
                confidence = int(box_split[1])
                rect = utils.Rectangle.from_yolo(box_split[2:])
                result[frame].append((name, confidence, rect))

        return result


    def annotate(self, img: np.ndarray, boxes: List[utils.Rectangle]):
        for box in boxes:
            rect = box[2]
            img = cv2.rectangle(
                img,
                rect.get_topleft_int(),
                rect.get_bottomright_int(),
                (0, 128, 255),
                3
            )
            img = cv2.putText(
                img,
                f'{box[0]}: {box[1]:02d}%',
                rect.get_topleft_int(),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 128, 255),
                2
            )


    def run_validation(self, sequence: str) -> None:
        midgard = Midgard(sequence)
        output = utils.get_output('output', midgard.orig_capture)
        try:
            img_input = midgard.img_path + '/image_%5d.png'
            video_path = midgard.seq_path + '/video.mp4'
            video_annotated_path = midgard.seq_path + '/video-annotated.mp4'
            self.img_to_video(img_input, video_path)

            frames = self.get_inference(video_path, video_annotated_path, False)
            frames = self.parse_frames(frames)
            print(len(frames), midgard.N)

            for i in range(midgard.N):
                frame = midgard.get_frame()
                if i in frames:
                    self.annotate(frame, frames[i])
                    output.write(frame)
        finally:
            output.release()


    def img_to_video(self, input: str, output: str):
        if not os.path.exists(output):
            command = f'ffmpeg -r 30 -i {input} -c:v libx264 -vf fps=30 -pix_fmt yuv420p {output} -y'.split(' ')
            subprocess.call(command)



validator = Validator()
validator.run_validation('countryside-natural/north-narrow')
