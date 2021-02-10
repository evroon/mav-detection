import utils
import cv2
import os
import numpy as np
import flow_vis

class Midgard:
    def __init__(self, sequence: str, debug_mode: bool) -> None:
        self.sequence = sequence
        self.debug_mode = debug_mode

        midgard_path = os.environ['MIDGARD_PATH']
        self.img_path = f'{midgard_path}/{sequence}/images'
        self.ann_path = f'{midgard_path}/{sequence}/annotation'
        self.orig_capture = cv2.VideoCapture(f'{self.img_path}/image_%5d.png')
        self.flow_capture = cv2.VideoCapture(f'{self.img_path}/output/color_coding/video/000000.flo.mp4')
        self.capture_size = utils.get_capture_size(self.orig_capture)
        self.flow_size = utils.get_capture_size(self.flow_capture)
        self.N = utils.get_frame_count(self.flow_capture)
        self.frame_columns, self.frame_rows = 1, 1

        if debug_mode:
            self.frame_columns, self.frame_rows = 4, 2

        capture_size = (self.capture_size[0] * self.frame_columns, self.capture_size[1] * self.frame_rows)
        self.output = utils.get_output('detection', capture_size=capture_size, is_grey=not debug_mode)
        self.i = 0
        self.start_frame = 100
        self.is_exiting = False

        if self.capture_size != self.flow_size:
            print(f'Note: original capture with size {self.capture_size} does not match flow, which has size {self.flow_size}')

        if self.N != utils.get_frame_count(self.orig_capture) - 1:
            print('Input counts: (images, flow fields):', utils.get_frame_count(self.orig_capture), self.N)
            raise ValueError('Input sizes do not match.')

    def print_details(self):
        print(utils.get_frame_count(self.orig_capture))
        print(utils.get_frame_count(self.flow_capture))
        print(utils.get_fps(self.orig_capture))
        print(utils.get_fps(self.flow_capture))

    def get_midgard_annotation(self):
        path = f'{self.ann_path}/annot_{self.i:05d}.csv'

        with open(path, 'r') as f:
            for line in f.readlines():
                values = line.split(',')
                values = [round(float(x)) for x in values]
                topleft = (values[1], values[2])
                ground_truth = utils.Rectangle(topleft, (values[3], values[4]))

        self.ground_truth = ground_truth

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

    def get_frame(self):
        s1, orig_frame = self.orig_capture.read()

        while self.i < self.start_frame:
            s1, orig_frame = self.orig_capture.read()
            self.i += 1

        if not s1:
            return None

        self.orig_frame = orig_frame
        self.flow_uv = self.get_flow_uv()
        self.flow_vis = self.get_flow_vis(self.flow_uv)
        self.get_flow_radial(self.flow_vis)
        self.get_midgard_annotation()

        return orig_frame

    def is_active(self):
        return self.i < self.N - 1 and not self.is_exiting

    def write(self, frame):
        self.output.write(frame)

        if self.debug_mode:
            cv2.imshow('output', frame)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                self.is_exiting = True

        self.i += 1

        if self.i % int(self.N / 10) == 0:
            print('{:.2f}'.format(self.i / self.N * 100) + '%', self.i, '/', self.N)

    def release(self):
        self.orig_capture.release()
        self.flow_capture.release()
        self.output.release()
        cv2.destroyAllWindows()
