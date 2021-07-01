import os

from numpy.core.numeric import zeros_like
from datasets.dataset import Dataset
import logging
import numpy as np
from typing import cast

class Experiment(Dataset):
    '''Helper functions for the TNO experiment dataset.'''

    def __init__(self, logger: logging.Logger, sequence: str) -> None:
        experiment_path = os.environ['EXPERIMENT_PATH']
        super().__init__(experiment_path, logger, sequence)

        self.gps_states_csv: np.ndarray = np.genfromtxt(f'{self.state_path}/vn_gps_log.csv', delimiter=',', skip_header=1)
        self.imu_states_csv: np.ndarray = np.genfromtxt(f'{self.state_path}/vn_imu_log.csv', delimiter=',', skip_header=1)

        self.gps_first_timestamp = self.gps_states_csv[0, 2]

        self.imu_start_offset = 0.5 # seconds. IMU data starts somewhat earlier than the video
        self.cropped_start_frame = 4 * 60 + 54 + self.imu_start_offset
        self.skip_frame_factor = 16
        self.duration = 15

        self.fps = (self.N+1) / self.duration
        self.cropped_end_frame = self.cropped_start_frame + self.duration

        self.start_gps_line = self.gps_first_timestamp + self.cropped_start_frame
        self.end_gps_line = self.cropped_start_frame + self.cropped_end_frame

        # Relate video frames to imu and gps timestamps.
        video_timestamps = np.linspace(0, self.duration, self.N)
        self.video_gps_indices = np.zeros_like(video_timestamps, dtype=np.uint16)
        self.video_imu_indices = np.zeros_like(video_timestamps, dtype=np.uint16)

        for i, v in enumerate(video_timestamps):
            time_diff_gps = np.abs(self.gps_states_csv[:, 2] - v - self.gps_states_csv[0, 2] - self.cropped_start_frame)
            time_diff_imu = np.abs(self.imu_states_csv[:, 2] - v - self.imu_states_csv[0, 2] - self.cropped_start_frame)

            self.video_gps_indices[i] = np.argmin(time_diff_gps)
            self.video_imu_indices[i] = np.argmin(time_diff_imu)


    def get_default_sequence(self) -> str:
        return 'moving-sample-low-fps'

    def get_gps_state(self, i:int) -> np.ndarray:
        return cast(np.ndarray, self.gps_states_csv[self.video_gps_indices[i], :])

    def get_imu_state(self, i:int) -> np.ndarray:
        return cast(np.ndarray, self.imu_states_csv[self.video_imu_indices[i], :])

    def get_angular_difference(self, first:int, second:int) -> np.ndarray:
        imu_index_start, imu_index_end = self.video_imu_indices[first], self.video_imu_indices[first + 1]
        acc_diff = np.zeros(3)

        if imu_index_end <= imu_index_start:
            return np.array([0, 0, 0])

        for i in range(imu_index_start, imu_index_end):
            dt = self.imu_states_csv[i, 2] - self.imu_states_csv[i-1, 2]
            acc_diff += self.imu_states_csv[i, 6:] * dt

        acc_diff /= (imu_index_end - imu_index_start)
        print(imu_index_start, imu_index_end, acc_diff)

        acc_diff = acc_diff[[1, 2, 0]]
        acc_diff[0] = -acc_diff[0]
        acc_diff[1] = -acc_diff[1]
        return acc_diff

    def get_delta_time(self, i:int) -> float:
        return self.skip_frame_factor / self.fps
