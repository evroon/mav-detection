import os

from numpy.core.numeric import zeros_like
from datasets.dataset import Dataset
import logging
import numpy as np

class Experiment(Dataset):
    '''Helper functions for the TNO experiment dataset.'''

    def __init__(self, logger: logging.Logger, sequence: str) -> None:
        experiment_path = os.environ['EXPERIMENT_PATH']
        super().__init__(experiment_path, logger, sequence)

        self.gps_states_csv = np.genfromtxt(f'{self.state_path}/vn_gps_log.csv', delimiter=',', skip_header=1)
        self.imu_states_csv = np.genfromtxt(f'{self.state_path}/vn_imu_log.csv', delimiter=',', skip_header=1)

        self.gps_first_timestamp = self.gps_states_csv[0, 0]

        self.cropped_start_frame = 4 * 60 + 54
        self.duration = 15

        self.fps = ((self.N+1) / self.duration)
        self.cropped_end_frame = self.cropped_start_frame + self.duration

        self.start_gps_line = self.gps_first_timestamp + self.cropped_start_frame
        self.end_gps_line = self.cropped_start_frame + self.cropped_end_frame

        # Relate video frames to imu and gps timestamps.
        video_timestamps = np.arange(0, self.N) / self.fps
        self.video_gps_indices = np.zeros_like(video_timestamps, dtype=np.uint16)
        self.video_imu_indices = np.zeros_like(video_timestamps, dtype=np.uint16)

        for i, v in enumerate(video_timestamps):
            time_diff_gps = self.gps_states_csv[:, 0] - v
            time_diff_imu = self.imu_states_csv[:, 0] - v

            self.video_gps_indices[i] = np.argmin(time_diff_gps)
            self.video_imu_indices[i] = np.argmin(time_diff_imu)

    def get_default_sequence(self) -> str:
        return 'moving-sample'

    def get_gps_state(self, i:int) -> np.ndarray:
        return self.gps_states_csv[self.video_gps_indices[i], :]

    def get_imu_state(self, i:int) -> np.ndarray:
        return self.imu_states_csv[self.video_imu_indices[i], :]

    def get_angular_difference(self, first:int, second:int) -> np.ndarray:
        imu_index_start, imu_index_end = self.video_imu_indices[first], self.video_imu_indices[second]
        acc_diff = np.zeros(3)

        for i in range(imu_index_start, imu_index_end):
            dt = self.imu_states_csv[i, 2] - self.imu_states_csv[i-1, 2]
            acc_diff += self.imu_states_csv[i, 6:] * dt

        return acc_diff

    def get_delta_time(self, i:int) -> float:
        return 1 / self.fps
