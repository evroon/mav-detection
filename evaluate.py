import requests
import utils
import cv2
import numpy as np
import utils
from midgard import Midgard

host: str = 'http://192.168.178.235:8099'


def get_inference(input_file: str, output_file: str, use_default_weights=False) -> bool:
    headers = {'accept': 'application/json'}
    params = (
        ('use_default_weights', use_default_weights),
    )
    files = {
        'video': (input_file, open(input_file, 'rb')),
    }

    response = requests.post(f'{host}/predict_video', headers=headers, params=params, files=files)
    open(output_file, 'wb').write(response.content)

    response = requests.get(f'{host}/predict_video_boxes')
    return response.json()

def parse_frames(frames: dict) -> dict:
    result = dict()
    for frame, boxes in frames.items():
        result[frame] = []
        for _, box in enumerate(boxes):
            box_split = box.split(' ')
            name = box_split[0]
            confidence = box_split[0]
            rect = utils.Rectangle.from_yolo(box_split[2:])
            result[frame].append((name, confidence, rect))

    return result

def annotate(img: np.ndarray):
    pass

def annotate_video(sequence: str):
    midgard = Midgard(sequence, True)
    frame = midgard.get_frame()
    annotate(frame)


frames = get_inference('media/V_AIRPLANE_002.mp4', 'media/V_AIRPLANE_002-out.mp4', True)
parse_frames(frames)
