import utils
import cv2
import os
import numpy as np
import flow_vis

class Midgard:
    def __init__(self, sequence):
        self.sequence = sequence

        kitti_path = os.environ['MIDGARD_PATH']
        self.img_path = f'{kitti_path}/{sequence}/images'
        self.ann_path = f'{kitti_path}/{sequence}/annotation'
        self.orig_capture = cv2.VideoCapture(f'{self.img_path}/image_%5d.png')
        self.flow_capture = cv2.VideoCapture('media/flownet2-sports-hall.mp4')
        self.capture_size = utils.get_capture_size(self.orig_capture)
        self.flow_size = utils.get_capture_size(self.flow_capture)
        self.N = utils.count_dir(self.img_path)
        self.output = utils.get_output('detection', capture_size=(self.capture_size[0] * 2, self.capture_size[1] * 2))
        self.i = 0

        if self.capture_size != self.flow_size:
            print(f'original capture has size {self.capture_size}, does not match flow, which has size {self.flow_size}')

    def print_details(self):
        print(utils.get_frame_count(orig_capture))
        print(utils.get_frame_count(flow_capture))
        print(utils.get_fps(orig_capture))
        print(utils.get_fps(flow_capture))

    def get_midgard_annotation(self):
        path = f'{self.ann_path}/annot_{self.i:05d}.csv'

        with open(path, 'r') as f:
            for line in f.readlines():
                values = line.split(',')
                values = [round(float(x)) for x in values]
                topleft = (values[1], values[2])
                rect = utils.Rectangle(topleft, (values[3], values[4]))

        return rect

    def get_flow_uv(self):
        flo_path = f'{self.img_path}/output/inference/run.epoch-0-flow-field/{self.i:06d}.flo'
        flow_uv = utils.read_flow(flo_path)

        if self.capture_size != self.flow_size:
            flow_uv = cv2.resize(flow_uv, self.capture_size)

        return flow_uv

    def get_flow_vis(self, frame):
        return flow_vis.flow_to_color(frame, convert_to_bgr=True)

    def get_flow_radial(self, frame):
        flow_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        flow_hsv[..., 0] = flow_hsv[..., 0]
        flow_hsv[..., 1] = 255
        flow_hsv[..., 2] = 255
        return cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)

    def get_fft(self, frame):
        fft = np.fft.fft2(frame[..., 0])
        fshift = np.fft.fftshift(fft)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        magnitude_rgb = np.zeros_like(frame)
        magnitude_rgb[..., 0] = magnitude_spectrum
        return magnitude_rgb

    def get_frame(self):
        s1, orig_frame = self.orig_capture.read()

        if not s1:
            return None

        return orig_frame

    def iterate(self):
        self.i += 1

        if self.i % int(self.N / 10) == 0:
            print('{:.2f}'.format(self.i / self.N * 100) + '%', self.i, '/', self.N)

    def is_active(self):
        return self.i < self.N - 1


    def write(self, frame):
        self.output.write(frame)

    def release(self):
        self.orig_capture.release()
        self.flow_capture.release()
        self.output.release()
        cv2.destroyAllWindows()
