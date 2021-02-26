import numpy as np
import cv2
import time
import sys

# Source: https://www.youtube.com/watch?v=uv73CjWscxI&ab_channel=SahakornBuangam
# code: https://github.com/sahakorn/Python-optical-flow-tracking/blob/master/optical_flow.py

count = 0


class Farneback:
    def __init__(self, capture: cv2.VideoCapture, output: cv2.VideoWriter) -> None:
        self.capture = capture
        self.output = output

        _, prev = self.capture.read()
        capture_size = prev.shape

        self.hsv = np.zeros_like(prev)
        self.prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        self.cur_glitch = prev.copy()

        self.history_length = 1
        self.prev_results = np.zeros((*capture_size, self.history_length))
        self.rolling_history_id = 0

    def draw_flow(self, img: np.ndarray, flow: np.ndarray, step: int = 8) -> np.ndarray:
        h, w = img.shape[:2]
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(np.int)
        fx, fy = flow[y, x].T

        threshold = 1.0

        mask = np.sqrt(fx ** 2.0 + fy ** 2.0) > threshold
        x = np.extract(mask, x)
        y = np.extract(mask, y)
        fx = np.extract(mask, fx)
        fy = np.extract(mask, fy)

        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        cv2.polylines(img, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
        return img


    def draw_hsv(self, flow: np.ndarray) -> np.ndarray:
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang*(180/np.pi/2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v*4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr


    def warp_flow(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = flow.shape[:2]
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res


    def process(self) -> np.ndarray:
        _, img = self.capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev=self.prevgray,
                                            next=gray, flow=None,
                                            pyr_scale=0.4, levels=1, winsize=12,
                                            iterations=10, poly_n=8, poly_sigma=1.2,
                                            flags=0)
        self.prevgray = gray

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv = np.zeros_like(img)
        self.hsv[:, :, 0] = ang*180/np.pi/2
        self.hsv[:, :, 1] = 255
        self.hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) * 2.0

        invalid_frame =  np.sum(self.hsv[:, :, 2]) < 1

        mask = self.hsv[:, :, 2] < 1
        self.hsv[mask, 0] = 127
        self.hsv[mask, 2] = 255

        result = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        if invalid_frame:
            result = self.prev_results[..., 0].astype(np.uint8)

        # result = (np.sum(self.prev_results, -1) + bgr) / (1 + self.history_length)
        # result = result.astype(np.uint8)
        self.prev_results[..., self.rolling_history_id] = result

        if self.rolling_history_id >= self.history_length:
            self.rolling_history_id = 0

        return result
