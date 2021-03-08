import airsim
import os
import pprint
import subprocess
import msgpackrpc
import json
import numpy as np
import math
import time

from datetime import datetime
from multiprocessing import Process
from typing import Dict, List, Any, Optional, cast

class AirSimControl:
    def __init__(self) -> None:
        self.observing_drone = 'Drone1'
        self.target_drone = 'Drone2'

        # self.radii = [1.0, 2.5, 5.0, 7.5, 10, 15, 20, 25, 30]
        self.radii = [5.0, 7.5, 10, 15, 20]
        self.altitude = 10
        self.speed = 2.0
        self.orbits = 1
        self.snapshots = 0
        self.snapshot_delta = None
        self.next_snapshot = None
        self.z = None
        self.snapshot_index = 0
        self.did_takeoff = False
        self.drones_offset_x = 5
        self.executable = r'D:\UnrealProjects\CityParkEnvironmentCollec\Builds\WindowsNoEditor\CityParkEnvironmentCollec.exe'
        self.running = True
        self.iteration: int = 0
        self.timestamps: Dict[int, datetime] = {}

    def init(self) -> None:
        # Connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        try:
            self.client.confirmConnection()
        except msgpackrpc.error.TransportError:
            self.start_sim()
            self.client.confirmConnection()

        self.client.simSetSegmentationObjectID("[\w]*", 0, True)
        self.client.simSetSegmentationObjectID("Drone[\w]*", 255, True)

        self.client.enableApiControl(True, self.observing_drone)
        self.client.enableApiControl(True, self.target_drone)
        self.client.armDisarm(True, self.observing_drone)
        self.client.armDisarm(True, self.target_drone)

        self.clean()

    def start_sim(self) -> None:
        subprocess.call([self.executable])

    def get_position(self, vehicle_name: str = None) -> airsim.Vector3r:
        if vehicle_name is None:
            vehicle_name = self.target_drone

        return self.client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated.position

    def takeoff(self) -> None:
        landed = self.client.getMultirotorState().landed_state
        if landed == airsim.LandedState.Landed:
            print('Takeoff...')
            takeoff1 = self.client.takeoffAsync(vehicle_name=self.observing_drone)
            takeoff2 = self.client.takeoffAsync(vehicle_name=self.target_drone)
            takeoff1.join()
            takeoff2.join()

        self.align_north()

        z = self.get_position(vehicle_name=self.observing_drone).z_val

        self.center = self.get_position(vehicle_name=self.observing_drone)
        self.client.moveToPositionAsync(self.center.x_val - self.radii[0], self.center.y_val, z, 2, vehicle_name=self.target_drone).join()

    def align_north(self) -> None:
        align1 = self.client.rotateToYawAsync(0, 1, vehicle_name=self.observing_drone)
        align2 = self.client.rotateToYawAsync(0, 1, vehicle_name=self.target_drone)
        align1.join()
        align2.join()

    def start_trajectory(self) -> None:
        f1 = self.client.moveByVelocityAsync(5, 0, 0, 20, vehicle_name=self.observing_drone)
        f2 = self.client.moveByVelocityAsync(5, 0, 0, 20, vehicle_name=self.target_drone)

    def fly_path(self) -> None:
        z = 0
        path = [
            airsim.Vector3r(50,     0,     z),
            airsim.Vector3r(80,    -80,    z),
            airsim.Vector3r(0,      0,     z)
        ]

        self.move_on_path = self.client.moveOnPathAsync(
                        path,
                        6, 120,
                        airsim.DrivetrainType.ForwardOnly,
                        airsim.YawMode(False, 0), 20, 1)

    def fly(self) -> None:
        for radius in self.radii:
            self.fly_orbit(radius)

    def fly_orbit(self, radius: float) -> None:
        count = 0
        self.start_angle: Optional[float] = None
        self.next_snapshot = None

        start = self.get_position()
        self.z = start.z_val

        # ramp up time
        ramptime = radius / 10
        self.start_time = time.time()

        print(f'Starting orbit with radius of {radius:0.2f}m')

        while count < self.orbits:
            if self.snapshots > 0 and not (self.snapshot_index < self.snapshots):
                break

            # ramp up to full speed in smooth increments so we don't start too aggressively.
            now = time.time()
            speed = self.speed
            diff = now - self.start_time
            if diff < ramptime:
                speed = self.speed * diff / ramptime
            elif ramptime > 0:
                # print("reached full speed...")
                ramptime = 0

            lookahead_angle = speed / radius

            pos = self.get_position()
            dx = pos.x_val - self.center.x_val
            dy = pos.y_val - self.center.y_val
            actual_radius = math.sqrt((dx*dx) + (dy*dy))
            angle_to_center = math.atan2(dy, dx)

            camera_heading = (angle_to_center - math.pi) * 180 / math.pi

            # compute lookahead
            lookahead_x = self.center.x_val + radius * math.cos(angle_to_center + lookahead_angle)
            lookahead_y = self.center.y_val + radius * math.sin(angle_to_center + lookahead_angle)

            vx = lookahead_x - pos.x_val
            vy = lookahead_y - pos.y_val

            if self.track_orbits(radius, angle_to_center * 180 / math.pi):
                count += 1
                print("Completed {} orbit(s)".format(count))

            self.camera_heading = camera_heading
            self.client.moveByVelocityZAsync(vx, vy, self.z, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading), vehicle_name=self.target_drone)

            # Capture frames
            self.get_states()

            responses = self.client.simGetImages([
                airsim.ImageRequest("high_res", airsim.ImageType.Scene),
                airsim.ImageRequest("segment", airsim.ImageType.Segmentation)
            ], vehicle_name=self.observing_drone)

            # Save images.
            for response in responses:
                if response.pixels_as_float:
                    airsim.write_pfm(os.path.normpath('depth.pfm'), airsim.get_pfm_array(response))
                else:
                    image_type: str = 'images' if response.image_type == airsim.ImageType.Scene else 'segmentations'
                    airsim.write_file(os.path.normpath(f'{image_type}/image_{self.iteration:05d}.png'), response.image_data_uint8)

            self.timestamps[self.iteration] = self.get_time()

            if self.iteration > 1:
                difference = self.timestamps[self.iteration] - self.timestamps[self.iteration - 1]
                # print(f'Elapsed time during iteration {self.iteration} in real-time: {difference.microseconds / 1000}ms')

            self.iteration += 1

    def track_orbits(self, radius: float, angle: float) -> bool:
        # tracking # of completed orbits is surprisingly tricky to get right in order to handle random wobbles
        # about the starting point.  So we watch for complete 1/2 orbits to avoid that problem.
        if angle < 0:
            angle += 360

        if self.start_angle is None:
            self.start_angle = angle
            if self.snapshot_delta:
                self.next_snapshot = angle + self.snapshot_delta
            self.previous_angle = angle
            self.shifted = False
            self.previous_sign = None
            self.previous_diff = None
            self.quarter = False
            return False

        # now we just have to watch for a smooth crossing from negative diff to positive diff
        if self.previous_angle is None:
            self.previous_angle = angle
            return False

        # ignore the click over from 360 back to 0
        if self.previous_angle > 350 and angle < 10:
            if self.snapshot_delta and self.next_snapshot >= 360:
                self.next_snapshot -= 360
            return False

        diff = self.previous_angle - angle
        crossing = False
        self.previous_angle = angle

        if self.snapshot_delta and angle > self.next_snapshot:
            print("Taking snapshot at angle {}".format(angle))
            self.take_snapshot()
            self.next_snapshot += self.snapshot_delta

        diff = abs(angle - self.start_angle)

        if diff > 45:
            self.quarter = True

        if self.quarter and self.previous_diff is not None and diff != self.previous_diff:
            # watch direction this diff is moving if it switches from shrinking to growing
            # then we passed the starting point.
            direction = 1.0 if self.previous_diff - diff > 0.0 else -1.0
            # print(direction, diff, self.snapshot)
            if self.previous_sign is None:
                self.previous_sign = direction
            elif self.previous_sign > 0 and direction < 0:
                if diff < 45:
                    self.quarter = False
                    if self.snapshots <= self.snapshot_index + 1:
                        crossing = True
            self.previous_sign = direction

        self.previous_diff = diff
        return crossing

    def get_magnitude(self, vector: np.ndarray) -> float:
        return float(np.sqrt(vector.x_val ** 2.0 + vector.y_val ** 2.0 + vector.z_val ** 2.0))

    def get_time_formatted(self, time: datetime= None) -> str:
        if time is None:
            time = self.get_time()

        return time.strftime("%H-%M-%S.%f")

    def get_time(self) -> datetime:
        return datetime.now()

    def get_json(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return cast(Dict[str, Any], json.loads(
            json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)))
        ))

    def get_states(self) -> None:
        state1 = self.client.getMultirotorState(vehicle_name=self.target_drone)
        # state2 = self.client.getMultirotorState(vehicle_name=self.target_drone)

        with open(f'states/{self.get_time_formatted()}.json', 'w') as f:
            f.write(json.dumps(self.get_json(state1), indent=4, sort_keys=True))

        if self.iteration > 10 and abs(self.get_magnitude(state1.kinematics_estimated.position)) < 2.0:
            print('stopped')
            self.running = False

    def clean(self) -> None:
        print('Removing previous results...')

        for f in os.listdir('images'):
            os.remove(f'images/{f}')

        for f in os.listdir('segmentations'):
            os.remove(f'segmentations/{f}')

        for f in os.listdir('states'):
            os.remove(f'states/{f}')

        for f in os.listdir('FOEs'):
            os.remove(f'FOEs/{f}')

    def land(self) -> None:
        print('landing...')
        self.client.moveToPositionAsync(self.center.x_val + 3, self.center.y_val, self.z, self.speed, vehicle_name=self.target_drone).join()

        self.align_north()

        f1 = self.client.landAsync(vehicle_name=self.observing_drone)
        f2 = self.client.landAsync(vehicle_name=self.target_drone)
        f1.join()
        f2.join()

    def finish(self) -> None:
        self.timestamps_str = {}

        if self.timestamps is not None:
            for idx, time in self.timestamps.items():
                self.timestamps_str[idx] = self.get_time_formatted(time)

            with open(f'states/timestamps.json', 'w') as f:
                f.write(json.dumps(self.timestamps_str, indent=4, sort_keys=True))

    def run(self) -> None:
        try:
            self.init()
            self.takeoff()
            self.fly()
            self.land()
        finally:
            self.finish()


control = AirSimControl()
control.run()
