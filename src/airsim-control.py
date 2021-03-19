import airsim
import os
import pprint
import subprocess
import msgpackrpc
import json
import numpy as np
import math
import time
import shutil
import argparse

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, cast

from sim_config import SimConfig
from run_config import RunConfig


class AirSimControl:
    def __init__(self, collection_name: str) -> None:
        self.observing_drone = 'Drone1'
        self.target_drone = 'Drone2'
        self.root_data_dir = 'data'
        self.collection_name = collection_name

        collection = RunConfig.get_settings()['collections'][self.collection_name]

        orientations_str = collection['orientations']
        locations = collection['locations']
        orbit_speeds = collection['orbit_speed']
        global_speeds = collection['global_speed']
        heights = collection['heights']
        radii = collection['radii']

        orientations = [SimConfig.get_orientation(x) for x in orientations_str]
        self.configs = []
        self.largest_radius = radii[-1]

        for sequence_name, center in locations.items():
            for orbit_speed in orbit_speeds:
                for global_speed_key, global_speed in global_speeds.items():
                    for height_name, height in heights.items():
                        for orientation in orientations:
                            for radius in radii:
                                config = SimConfig(
                                    sequence_name,
                                    height_name,
                                    airsim.Vector3r(center['x'], center['y'], center['z'] - height),
                                    orientation,
                                    radius,
                                    center['z'],
                                    orbit_speed,
                                    airsim.Vector3r(global_speed['lin_x'], global_speed['sin_y'], global_speed['sin_z']),
                                    global_speed_key
                                )
                                if not os.path.exists(self.get_base_dir(config)):
                                    self.configs.append(config)
                                else:
                                    print(f'Skipping {config.full_name()}')

        print(f'Number of locations: {len(locations)}')
        print(f'Number of configurations: {len(self.configs)}')

        self.base_velocity: Tuple[float, float] = (0, 0)
        self.speed = 3.0
        self.snapshot_index = 0
        self.did_takeoff = False
        self.drones_offset_x = 5
        self.iteration: int = 0
        self.timestamps: Dict[int, datetime] = {}
        self.begin_time: datetime = datetime.now()
        self.direction = 1
        self.drone_in_frame_previous = False

        if not os.path.exists(self.root_data_dir + '/states'):
            os.makedirs(self.root_data_dir + '/states')

    def init(self) -> None:
        # Connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        try:
            print('Connecting...')
            self.client.confirmConnection()
        except msgpackrpc.error.TransportError:
            print('AirSim is not running or could not connect.')
            exit(-1)

        self.client.simSetSegmentationObjectID("[\w]*", 0, True)
        self.client.simSetSegmentationObjectID("Drone[\w]*", 255, True)

        self.client.enableApiControl(True, self.observing_drone)
        self.client.enableApiControl(True, self.target_drone)
        self.client.armDisarm(True, self.observing_drone)
        self.client.armDisarm(True, self.target_drone)

        self.clean()

    def get_position(self, vehicle_name: str = None) -> airsim.MultirotorState:
        if vehicle_name is None:
            vehicle_name = self.target_drone

        return self.client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated.position

    def is_landed(self, vehicle_name: str) -> bool:
        return cast(bool, self.client.getMultirotorState(vehicle_name=vehicle_name).landed_state == airsim.LandedState.Landed)

    def takeoff(self, vehicle_name: Optional[str] = None) -> Optional[msgpackrpc.future.Future]:
        if vehicle_name is None:
            print('Takeoff...')
            takeoff1 = self.takeoff(self.observing_drone)
            takeoff2 = self.takeoff(self.target_drone)

            if takeoff1 is not None:
                takeoff1.join()

            if takeoff2 is not None:
                takeoff2.join()
        else:
            if self.is_landed(vehicle_name):
                return self.client.takeoffAsync(vehicle_name=vehicle_name)

        return None

    def align_north(self) -> None:
        align1 = self.client.rotateToYawAsync(0, 2, vehicle_name=self.observing_drone)
        align2 = self.client.rotateToYawAsync(0, 2, vehicle_name=self.target_drone)
        align1.join()
        align2.join()

    def move_to_position(self, config: SimConfig, vehicle_name: str, z: float = None) -> msgpackrpc.future.Future:
        position = config.get_start_position(vehicle_name == self.observing_drone)

        if z is not None:
            position.z_val = z

        return self.client.moveToPositionAsync(position.x_val, position.y_val, position.z_val, self.speed, vehicle_name=vehicle_name)

    def fly(self) -> None:
        for i, config in enumerate(self.configs):
            first_sequence_of_kind = i == 0 or config.is_different_location(self.configs[i-1])
            first_sequence_of_pose = i == 0 or config.is_different_pose(self.configs[i-1])
            first_sequence_of_height = i == 0 or config.is_different_height(self.configs[i-1])
            will_teleport = i == 0 or i >= len(self.configs) - 1 or config.is_different_location(self.configs[i+1])
            last_sequence_of_kind = i >= len(self.configs) - 1 or config.is_different(self.configs[i+1])

            if first_sequence_of_kind:
                self.prepare_run(config)
            elif first_sequence_of_height:
                print(f'Moving drone to height: {config.center.z_val:.02f}')
                f1 = self.move_to_position(config, self.observing_drone)
                f2 = self.move_to_position(config, self.target_drone)
                f1.join()
                f2.join()
                self.teleport(config)
            elif first_sequence_of_pose:
                print(f'Rotating drone to: {config.orientation}')
                self.teleport(config)

            self.fly_orbit(config)

            if last_sequence_of_kind:
                self.finish_sequence()

            if will_teleport:
                self.client.armDisarm(False, self.observing_drone)
                self.client.armDisarm(False, self.target_drone)
                self.wait_for_landing()

    def teleport(self, config: SimConfig) -> None:
        heading = config.orientation.get_heading() / 180.0 * np.pi

        # Move observing drone to center of orbit.
        quat = airsim.to_quaternion(0.0, 0.0, heading)
        pose = airsim.Pose(config.get_start_position(True), quat)
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.observing_drone)

        # Move target drone to start point in orbit.
        quat = airsim.to_quaternion(0.0, 0.0, 0.0)
        pose = airsim.Pose(config.get_start_position(False), quat)
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.target_drone)

    def wait_for_landing(self) -> None:
        old_pos = airsim.Vector3r()
        while (self.get_position(self.observing_drone) - old_pos).get_length() > 0.01:
            old_pos = self.get_position(self.observing_drone)
            time.sleep(1)

    def prepare_run(self, config: SimConfig) -> None:
        self.teleport(config)
        self.wait_for_landing()

        self.client.armDisarm(True, self.observing_drone)
        self.client.armDisarm(True, self.target_drone)
        self.takeoff()
        self.teleport(config)

        print(f'{config.base_name}: New heading: {config.orientation}, altitude: {config.center.z_val:.02f}m')

    def get_base_dir(self, config: SimConfig) -> str:
        return f'{self.root_data_dir}/{config}'

    def capture(self) -> bool:
        # Capture frames.
        responses = self.client.simGetImages([
            airsim.ImageRequest("segment", airsim.ImageType.Segmentation),
            airsim.ImageRequest("high_res", airsim.ImageType.Scene),
        ], vehicle_name=self.observing_drone)

        # Save images.
        for response in responses:
            if response.pixels_as_float:
                airsim.write_pfm(os.path.normpath('depth.pfm'), airsim.get_pfm_array(response))
            else:
                image_type: str = 'images' if response.image_type == airsim.ImageType.Scene else 'segmentations'
                image_path = f'{self.base_dir}/{image_type}/image_{self.iteration:05d}.png'
                seg_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

                if response.image_type == airsim.ImageType.Segmentation:
                    # 343392 is needed for a 800x600 image because the image is compressed,
                    # would be zero is image is uncompressed, see airsim.ImageRequest.
                    drone_in_frame = np.sum(seg_1d) > 343392 and self.iteration > 10
                    if drone_in_frame:
                        airsim.write_file(os.path.normpath(image_path), response.image_data_uint8)
                    elif self.drone_in_frame_previous and self.iteration > 30:
                        return False

                    self.drone_in_frame_previous = drone_in_frame
                elif self.drone_in_frame_previous:
                    airsim.write_file(os.path.normpath(image_path), response.image_data_uint8)
                    self.timestamps[self.iteration] = self.get_time()

        if self.drone_in_frame_previous:
            self.write_states()

        return True

    def fly_orbit(self, config: SimConfig) -> None:
        print(f'Starting orbit with radius of {config.radius:0.2f}m')

        self.start_angle: Optional[float] = None
        self.base_dir = self.get_base_dir(config)
        self.drone_in_frame_previous = False
        self.direction *= -1
        running = True
        lookahead_angle = config.orbit_speed * np.pi / 180.0 * self.direction

        self.prepare_sequence()

        while running:
            pos_target_drone = self.get_position()
            pos_observer_drone = self.get_position(self.observing_drone)

            dx = pos_target_drone.x_val - pos_observer_drone.x_val
            dy = pos_target_drone.y_val - pos_observer_drone.y_val
            angle_to_center = math.atan2(dy, dx)
            camera_heading = (angle_to_center - math.pi) * 180 / math.pi

            # compute lookahead
            lookahead_x = pos_observer_drone.x_val + config.radius * math.cos(angle_to_center + lookahead_angle)
            lookahead_y = pos_observer_drone.y_val + config.radius * math.sin(angle_to_center + lookahead_angle)

            vx = lookahead_x - pos_target_drone.x_val + config.global_speed.x_val
            vy = lookahead_y - pos_target_drone.y_val
            vz = pos_observer_drone.z_val

            self.client.moveByVelocityZAsync(vx, vy, vz, 1, airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, camera_heading), vehicle_name=self.target_drone)

            self.client.moveByVelocityZAsync(config.global_speed.x_val, config.global_speed.y_val, config.center.z_val,
                1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(), vehicle_name=self.observing_drone)

            running = self.capture()
            self.iteration += 1

    def get_magnitude(self, vector: np.ndarray) -> float:
        return float(np.sqrt(vector.x_val ** 2.0 + vector.y_val ** 2.0 + vector.z_val ** 2.0))

    def get_time_formatted(self, time: datetime = None) -> str:
        if time is None:
            time = self.get_time()

        return f'{time.timestamp() * 1000:.0f}'

    def get_time(self) -> datetime:
        return datetime.now()

    def get_json(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return cast(Dict[str, Any], json.loads(
            json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)))
        ))

    def write_states(self) -> None:
        state1 = self.client.getMultirotorState(vehicle_name=self.target_drone)

        with open(f'{self.base_dir}/states/{self.get_time_formatted()}.json', 'w') as f:
            f.write(json.dumps(self.get_json(state1), indent=4, sort_keys=True))

    def clean(self) -> None:
        print('Removing previous results of states...')
        max_attempts = 10
        i = 0

        while i < max_attempts:
            try:
                for f in os.listdir(f'{self.root_data_dir}/states'):
                    os.remove(f'{self.root_data_dir}/states/{f}')
            finally:
                i += 1
            break

    def land(self, config: SimConfig) -> None:
        print('Landing...')
        f1 = self.move_to_position(config, self.observing_drone, z=config.ground_height - 1.0)
        f2 = self.move_to_position(config, self.target_drone, z=config.ground_height - 1.0)
        f1.join()
        f2.join()

        f1 = self.client.landAsync(vehicle_name=self.observing_drone)
        f2 = self.client.landAsync(vehicle_name=self.target_drone)
        f1.join()
        f2.join()

    def finish_sequence(self) -> None:
        self.timestamps_str = {}

        if self.timestamps is not None:
            for idx, time in self.timestamps.items():
                self.timestamps_str[idx] = self.get_time_formatted(time)

            with open(f'{self.base_dir}/states/timestamps.json', 'w') as f:
                f.write(json.dumps(self.timestamps_str, indent=4, sort_keys=True))
                self.timestamps = {}

        self.link_ue4_output()

    def link_ue4_output(self) -> None:
        """
        Link the output of ue4 json files to the json files generated in self.get_states()
        according to their timestamp.
        """
        states_in_dir = f'{self.root_data_dir}/states'
        states_out_dir = f'{self.base_dir}/states'

        def list_timestamps(states_dir: str) -> Tuple[List[str], np.ndarray]:
            files = os.listdir(states_dir)
            files.sort()
            timestamps = [int(os.path.basename(x).rstrip('.json')) for x in files if 'timestamp' not in x]
            timestamps = np.array(timestamps)
            return [f'{states_dir}/{x}' for x in files], timestamps

        in_files, in_timestamps = list_timestamps(states_in_dir)
        out_files, out_timestamps = list_timestamps(states_out_dir)

        for out_file, out_timestamp in zip(out_files, out_timestamps):
            diffs = in_timestamps - out_timestamp
            selected_input = np.argmin(np.abs(diffs))

            with open(out_file, 'r') as f_out:
                result = json.load(f_out)

            with open(in_files[selected_input], 'r') as f_in:
                with open(out_file, 'w') as f_out:
                    result['ue4'] = json.load(f_in)
                    result['thread_difference'] = int(diffs[selected_input])
                    f_out.write(json.dumps(result, indent=4, sort_keys=True))

    def create_if_not_exists(self, dir: str) -> None:
        if not os.path.exists(dir):
            os.makedirs(dir)

    def prepare_sequence(self) -> None:
        self.create_if_not_exists(f'{self.base_dir}/images')
        self.create_if_not_exists(f'{self.base_dir}/segmentations')
        self.create_if_not_exists(f'{self.base_dir}/states')

    def run(self) -> None:
        self.init()
        self.fly()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects MAVs in the dataset using optical flow.')
    parser.add_argument('--collection', type=str, help='collection to process', default='moving')
    args = parser.parse_args()

    control = AirSimControl(args.collection)
    control.run()

    scp_command = [
        'scp',
        '-r',
        'data',
        'erik@192.168.178.235:~/tno/datasets'
    ]
    subprocess.call(scp_command)
