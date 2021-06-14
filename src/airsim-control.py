import airsim
import os
import subprocess
import json
import numpy as np
import math
import time
import argparse
from dotenv import load_dotenv

from msgpackrpc.error import TransportError
from msgpackrpc.future import Future
from datetime import datetime
from scipy.spatial.transform import Rotation
from typing import Dict, List, Any, Optional, Tuple, cast

import utils
from sim_config import Mode, SimConfig
from run_config import RunConfig


class AirSimControl:
    def __init__(self, collection_name: str) -> None:
        self.observing_drone = 'Drone1'
        self.target_drone = 'Drone2'
        self.root_data_dir = 'data'
        self.collection_name = collection_name
        self.configs = []
        self.speed = 3.0
        self.iteration: int = 0
        self.timestamps: Dict[int, datetime] = {}
        self.direction = 1
        self.drone_in_frame_previous = False
        self.minimum_segmentation_sum = 1e12
        self.yaw_rate = 0 # deg/s
        self.max_yaw = np.deg2rad(30)
        self.delta_time = 0.033

        collection = RunConfig.get_settings()['collections'][self.collection_name]

        orientations_str = collection['orientations']
        locations = collection['locations']
        orbit_speeds = collection['orbit_speed']
        global_speeds = collection['global_speed']
        heights = collection['heights']
        radii = collection['radii']
        modes_str = collection['modes']
        collision_angles = collection['collision_angles']

        orientations = [SimConfig.get_orientation(x) for x in orientations_str]
        modes = [SimConfig.get_mode(x) for x in modes_str]

        for sequence_name, center in locations.items():
            for orbit_speed in orbit_speeds:
                for global_speed_key, global_speed in global_speeds.items():
                    for height_name, height in heights.items():
                        for orientation in orientations:
                            for radius in radii:
                                for mode in modes:
                                    for collision_angle in collision_angles:
                                        config = SimConfig(
                                            sequence_name,
                                            height_name,
                                            airsim.Vector3r(center['x'], center['y'], center['z'] - height),
                                            orientation,
                                            radius,
                                            center['z'],
                                            orbit_speed,
                                            airsim.Vector3r(global_speed['lin_x'], global_speed['sin_y'], global_speed['sin_z']),
                                            global_speed_key,
                                            mode,
                                            collision_angle
                                        )
                                        if not os.path.exists(self.get_base_dir(config)):
                                            self.configs.append(config)
                                        else:
                                            print(f'Skipping {config}')

        print(f'Number of locations: {len(locations)}')
        print(f'Number of configurations: {len(self.configs)}')

        if not os.path.exists(self.root_data_dir + '/states'):
            os.makedirs(self.root_data_dir + '/states')

    def init(self) -> None:
        """Initialize the AirSim connection."""
        # Connect to the AirSim simulator
        is_starting = False

        while True:
            try:
                print('Connecting...')
                self.client = airsim.MultirotorClient(ip=os.getenv('IP_ADDRESS'))
                self.client.confirmConnection()
                break
            except TransportError:
                time.sleep(1)

        self.client.simSetSegmentationObjectID("[\w]*", 0, True)
        self.client.simSetSegmentationObjectID("Drone[\w]*", 255, True)

        self.client.enableApiControl(True, self.observing_drone)
        self.client.enableApiControl(True, self.target_drone)
        self.client.armDisarm(True, self.observing_drone)
        self.client.armDisarm(True, self.target_drone)

        self.clean()
        self.prev_time = utils.get_time()

    def get_position(self, vehicle_name: str = None) -> airsim.Vector3r:
        """Get the position of an MAV.

        Args:
            vehicle_name (str, optional): the name of the vehicle to get the position of. Defaults to None.

        Returns:
            airsim.Vector3r: the position in meters.
        """
        if vehicle_name is None:
            vehicle_name = self.target_drone

        return self.client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated.position

    def get_yaw(self, vehicle_name: str = None) -> float:
        """Returns the yaw of a vehicle in radians.

        Args:
            vehicle_name (str, optional): the vehicle to get the yaw angle of. Defaults to None.

        Returns:
            float: the yaw angle in radians
        """
        if vehicle_name is None:
            vehicle_name = self.target_drone

        orientatation = self.client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated.orientation
        euler = Rotation.from_quat([orientatation.x_val, orientatation.y_val, orientatation.z_val, orientatation.w_val]).as_euler('xyz', degrees=False)
        return cast(float, euler[2])

    def is_landed(self, vehicle_name: str) -> bool:
        """Whether an MAV has landed."""
        return cast(bool, self.client.getMultirotorState(vehicle_name=vehicle_name).landed_state == airsim.LandedState.Landed)

    def takeoff(self, vehicle_name: Optional[str] = None) -> Optional[Future]:
        """Performs a takeoff for an MAV.

        Args:
            vehicle_name (Optional[str], optional): the name of the vehicle that will takeoff. Defaults to None.

        Returns:
            Optional[Future]: the async future result
        """
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
        """Align both drones to the north."""
        align1 = self.client.rotateToYawAsync(0, 2, vehicle_name=self.observing_drone)
        align2 = self.client.rotateToYawAsync(0, 2, vehicle_name=self.target_drone)
        align1.join()
        align2.join()

    def move_to_position(self, config: SimConfig, vehicle_name: str, z: float = None) -> Future:
        """Moves a vehicle to its new start position.

        Args:
            config (SimConfig): the configuration specifying the start position.
            vehicle_name (str): the name of the vehicle that will move
            z (float, optional): the altitude of the drone in meters. Defaults to None.

        Returns:
            Future: the result of the async move method
        """
        position = config.get_start_position(vehicle_name == self.observing_drone)

        if z is not None:
            position.z_val = z

        return self.client.moveToPositionAsync(position.x_val, position.y_val, position.z_val, self.speed, vehicle_name=vehicle_name)

    def fly(self) -> None:
        """Runs the control and data acquisition loop."""
        for config in self.configs:
            self.prepare_run(config)

            if config.mode == Mode.ORBIT:
                self.fly_orbit(config)
            elif config.mode == Mode.COLLISION:
                self.fly_collision(config)
            else:
                self.fly_straight(config)

            self.finish_sequence()

            self.client.armDisarm(False, self.observing_drone)
            self.client.armDisarm(False, self.target_drone)
            time.sleep(1)
            self.wait_for_landing()

    def teleport(self, config: SimConfig) -> None:
        """Teleport drones to the start locations of a new configuration.

        Args:
            config (SimConfig): the configuration specifying the start positions.
        """
        heading = np.deg2rad(config.orientation.get_heading())

        # Move observing drone to center of orbit.
        quat = airsim.to_quaternion(0.0, 0.0, heading)
        pose = airsim.Pose(config.get_start_position(True), quat)
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.observing_drone)

        # Move target drone to start point in orbit.
        quat = airsim.to_quaternion(0.0, 0.0, 0.0)
        pose = airsim.Pose(config.get_start_position(False), quat)
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.target_drone)

    def wait_for_landing(self) -> None:
        """Wait until the observing drone has landed."""
        print('Waiting to land...')
        old_pos = airsim.Vector3r()
        while (self.get_position(self.observing_drone) - old_pos).get_length() > 0.01:
            old_pos = self.get_position(self.observing_drone)
            time.sleep(1)

    def prepare_run(self, config: SimConfig) -> None:
        """Prepares the simulation for a new run.

        Args:
            config (SimConfig): the configuration to run.
        """
        self.teleport(config)
        self.wait_for_landing()

        self.client.armDisarm(True, self.observing_drone)
        self.client.armDisarm(True, self.target_drone)
        self.takeoff()
        self.teleport(config)

        print(f'{config.base_name}: New heading: {config.orientation}, altitude: {config.center.z_val:.02f}m')

    def get_base_dir(self, config: SimConfig) -> str:
        """Returns the base directory of the given configuration."""
        return f'{self.root_data_dir}/{config}'

    def write_frame(self, image_path: str, response: airsim.ImageResponse) -> None:
        """Writes an image response to disk

        Args:
            image_path (str): the output path
            response (airsim.ImageResponse): the image response containing the image
        """
        if response.pixels_as_float:
            pfm_array = airsim.get_pfm_array(response)
            airsim.write_pfm(os.path.normpath(image_path), pfm_array)
        else:
            airsim.write_file(os.path.normpath(image_path), response.image_data_uint8)

    def capture(self, config: SimConfig) -> None:
        """Capture frames"""
        responses = self.client.simGetImages([
            airsim.ImageRequest("segment", airsim.ImageType.Segmentation),
            airsim.ImageRequest("high_res", airsim.ImageType.Scene),
            airsim.ImageRequest("depth", airsim.ImageType.DepthPerspective, True),
        ], vehicle_name=self.observing_drone)

        self.prev_time = utils.get_time()

        # Save images.
        for response in responses:
            image_type: str = {
                airsim.ImageType.Scene: 'images',
                airsim.ImageType.DepthPerspective: 'depths',
                airsim.ImageType.Segmentation: 'segmentations',
            }[response.image_type]
            extension = 'pfm' if response.pixels_as_float else 'png'
            image_path: str = f'{self.base_dir}/{image_type}/image_{self.iteration:05d}.{extension}'

            if response.image_type == airsim.ImageType.Segmentation:
                # Determine the sum of seg_1d for a complete black mask.
                seg_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                seg_sum = np.sum(seg_1d)

                if self.minimum_segmentation_sum > seg_sum:
                    self.minimum_segmentation_sum = seg_sum

                drone_in_frame = config.mode == Mode.COLLISION or (seg_sum > self.minimum_segmentation_sum and self.iteration > 10)

                if drone_in_frame:
                    self.write_frame(image_path, response)

                self.drone_in_frame_previous = drone_in_frame
            elif self.drone_in_frame_previous:
                self.write_frame(image_path, response)
                self.timestamps[self.iteration] = utils.get_time()

                if response.image_type == airsim.ImageType.DepthPerspective and self.iteration > 10:
                    depth_img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    depth_std = np.std(depth_img)

                    if depth_std < 10:
                        raise ValueError(f'Depth perspective buffer probably incorrect, std of {depth_std} too small.')

        if self.drone_in_frame_previous:
            self.write_states()

    def fly_collision(self, config: SimConfig) -> None:
        """Let the drones fly on a collision course towards the same point with the same arrival time.

        Args:
            config (SimConfig): the configuration of the collision course
        """
        print(f'Starting collision course with length of {config.radius:0.2f}m')

        self.start_angle = None
        self.base_dir = self.get_base_dir(config)
        self.drone_in_frame_previous = False
        running = True

        self.prepare_sequence()

        pos_observer_drone = self.get_position(self.observing_drone)
        z = pos_observer_drone.z_val

        while running:
            # Continue for one timestap (1 second in real time, 1 second * clockspeed in simulation time).
            self.client.simContinueForTime(1)
            self.client.simPause(True)

            pos_target_drone = self.get_position()
            pos_observer_drone = self.get_position(self.observing_drone)

            self.client.moveToPositionAsync(config.center.x_val, config.center.y_val, z, config.global_speed.x_val, 100, airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(), vehicle_name=self.observing_drone)

            self.client.moveToPositionAsync(config.center.x_val, config.center.y_val, z, config.global_speed.x_val, 100, airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(), vehicle_name=self.target_drone)

            # Stop if drones are close enough to eachother. (2m)
            if (pos_target_drone - pos_observer_drone).get_length() < 2:
                running = False
                self.client.simPause(False)

            self.capture(config)
            self.iteration += 1

    def fly_straight(self, config: SimConfig) -> None:
        """Let the target drone fly in a straight line.

        Args:
            config (SimConfig): the configuration of the orbit
        """
        print(f'Starting line with distance of {config.radius:0.2f}m')

        self.base_dir = self.get_base_dir(config)
        self.drone_in_frame_previous = False
        running = True
        yaw_rate_direction = 1

        self.prepare_sequence()

        while running:
            pos_target_drone = self.get_position()
            pos_observer_drone = self.get_position(self.observing_drone)

            dx = pos_target_drone.x_val - pos_observer_drone.x_val
            dy = pos_target_drone.y_val - pos_observer_drone.y_val
            angle_to_center = math.atan2(dy, dx)
            camera_heading = np.rad2deg(angle_to_center)

            # Factor of 0.99333 is needed because the target drone will otherwise drift away from observer.
            vx = config.global_speed.x_val * 0.99333
            vy = config.orbit_speed * config.radius
            z = pos_observer_drone.z_val
            z_target = z - 0.15 * config.radius

            self.client.moveByVelocityZAsync(vx, vy, z_target, 10, airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, camera_heading), vehicle_name=self.target_drone)

            yaw_mode = airsim.YawMode(True, self.yaw_rate * yaw_rate_direction)

            self.client.moveByVelocityZAsync(config.global_speed.x_val, config.global_speed.y_val, config.center.z_val,
                10, airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode, vehicle_name=self.observing_drone)

            # Continue for one timestap (1 second in real time, 1 second * clockspeed in simulation time).
            self.client.simContinueForTime(1)
            self.client.simPause(True)
            self.capture(config)

            running = pos_target_drone.y_val < config.radius
            self.iteration += 1

    def fly_orbit(self, config: SimConfig) -> None:
        """Let the target drone fly an orbit.

        Args:
            config (SimConfig): the configuration of the orbit
        """
        print(f'Starting orbit with radius of {config.radius:0.2f}m')

        self.start_angle = None
        self.base_dir = self.get_base_dir(config)
        self.drone_in_frame_previous = False
        running = True
        lookahead_angle = config.orbit_speed * np.pi / 180.0 * self.direction
        yaw_rate_direction = 1

        self.prepare_sequence()

        while running:
            pos_target_drone = self.get_position()
            pos_observer_drone = self.get_position(self.observing_drone)

            dx = pos_target_drone.x_val - pos_observer_drone.x_val
            dy = pos_target_drone.y_val - pos_observer_drone.y_val
            angle_to_center = math.atan2(dy, dx)
            camera_heading = np.rad2deg(angle_to_center - math.pi)

            # compute lookahead
            lookahead_x = pos_observer_drone.x_val + config.radius * math.cos(angle_to_center + lookahead_angle)
            lookahead_y = pos_observer_drone.y_val + config.radius * math.sin(angle_to_center + lookahead_angle)

            vx = lookahead_x - pos_target_drone.x_val + config.global_speed.x_val
            vy = lookahead_y - pos_target_drone.y_val
            z = pos_observer_drone.z_val

            self.client.moveByVelocityZAsync(vx, vy, z, 10, airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, camera_heading), vehicle_name=self.target_drone)

            yaw_mode = airsim.YawMode(True, self.yaw_rate * yaw_rate_direction)

            self.client.moveByVelocityZAsync(config.global_speed.x_val, config.global_speed.y_val, config.center.z_val,
                10, airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode, vehicle_name=self.observing_drone)

            # Continue for one timestap (1 second in real time, 1 second * clockspeed in simulation time).
            self.client.simContinueForTime(1)
            self.client.simPause(True)

            base_heading = np.deg2rad(config.orientation.get_heading())
            angle_diff = self.get_yaw(self.observing_drone) - base_heading
            if abs(angle_diff) > self.max_yaw:
                yaw_rate_direction = -angle_diff / abs(angle_diff)

            self.capture(config)
            angle_diff = np.rad2deg(angle_to_center - base_heading)
            running = angle_diff < 50
            self.iteration += 1


    def get_time_formatted(self, time: datetime = None) -> str:
        """Get a formatted string of a given time

        Args:
            time (datetime, optional): the time to process. Defaults to None.

        Returns:
            str: the timestamp in ns.
        """
        if time is None:
            time = utils.get_time()

        return f'{time.timestamp() * 1000:.0f}'

    def write_states(self) -> None:
        """Write states of the vehicles to disk."""
        result: Dict[str, Any] = {}

        for vehicle_name in [self.observing_drone, self.target_drone]:
            state = self.client.getMultirotorState(vehicle_name=vehicle_name)
            imu_data = self.client.getImuData(imu_name="Imu", vehicle_name=vehicle_name)

            result[vehicle_name] = utils.get_json(state)
            result[vehicle_name]['imu'] = utils.get_json(imu_data)

        with open(f'{self.base_dir}/states/{self.get_time_formatted()}.json', 'w') as f:
            f.write(json.dumps(result, indent=4, sort_keys=True))

    def clean(self) -> None:
        """Clean results of previous runs."""
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
        """Land both drones.

        Args:
            config (SimConfig): the configuration specifying the ground height and land location
        """
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
        """Finish a sequence and write timestamps to disk."""
        print('Finishing sequence...')
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
            timestamps = np.array(
                [int(os.path.basename(x).rstrip('.json')) for x in files if 'timestamp' not in x]
            )
            return [f'{states_dir}/{x}' for x in files], timestamps

        in_files, in_timestamps = list_timestamps(states_in_dir)
        out_files, out_timestamps = list_timestamps(states_out_dir)

        if len(in_files) < 1:
            print('No input json files found.')
            return

        for out_file, out_timestamp in zip(out_files, out_timestamps):
            diffs = in_timestamps - out_timestamp
            selected_input = int(np.argmin(np.abs(diffs)))

            with open(out_file, 'r') as f_out:
                result = json.load(f_out)

            with open(in_files[selected_input], 'r') as f_in:
                with open(out_file, 'w') as f_out:
                    json_in = json.load(f_in)

                    for vehicle_name in [self.observing_drone, self.target_drone]:
                        result[vehicle_name]['ue4'] = json_in[vehicle_name]

                    result['thread_difference'] = int(diffs[selected_input])
                    f_out.write(json.dumps(result, indent=4, sort_keys=True))

    def prepare_sequence(self) -> None:
        utils.create_if_not_exists(f'{self.base_dir}/images')
        utils.create_if_not_exists(f'{self.base_dir}/segmentations')
        utils.create_if_not_exists(f'{self.base_dir}/depths')
        utils.create_if_not_exists(f'{self.base_dir}/states')

    def run(self) -> None:
        self.init()

        try:
            self.fly()
        finally:
            self.client.simPause(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects MAVs in the dataset using optical flow.')
    parser.add_argument('--collection',  type=str, help='collection to process', default='collision')
    parser.add_argument('--upload-only', action='store_true', help='upload images only')
    args = parser.parse_args()
    load_dotenv()

    if not args.upload_only:
        control = AirSimControl(args.collection)
        control.run()

    # scp_command = [
    #     'scp',
    #     '-r',
    #     'data',
    #     'erik@192.168.178.235:~/tno/datasets'
    # ]
    # subprocess.call(scp_command)
