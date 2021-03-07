import airsim
import os
import pprint
import subprocess
import msgpackrpc
import json
import numpy as np
from datetime import datetime
from multiprocessing import Process
from typing import Dict, List, Any, cast

class AirSimControl:
    def __init__(self) -> None:
        self.observing_drone = 'Drone1'
        self.target_drone = 'Drone2'

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

        self.executable = r'D:\UnrealProjects\CityParkEnvironmentCollec\Builds\WindowsNoEditor\CityParkEnvironmentCollec.exe'

    def start_sim(self) -> None:
        subprocess.call([self.executable])

    def takeoff(self) -> None:
        takeoff1 = self.client.takeoffAsync(vehicle_name=self.observing_drone)
        takeoff2 = self.client.takeoffAsync(vehicle_name=self.target_drone)
        takeoff1.join()
        takeoff2.join()

        self.align_north()

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

    def get_magnitude(self, vector: np.ndarray) -> float:
        return float(np.sqrt(vector.x_val ** 2.0 + vector.y_val ** 2.0 + vector.z_val ** 2.0))

    def get_time_formatted(self) -> str:
        return datetime.now().strftime("%H-%M-%S.%f")

    def get_json(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return cast(Dict[str, Any], json.loads(
            json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)))
        ))

    def get_states(self) -> None:
        state1 = self.client.getMultirotorState(vehicle_name=self.observing_drone)
        # state2 = self.client.getMultirotorState(vehicle_name=self.target_drone)


        with open(f'states/{self.get_time_formatted()}.json', 'w') as f:
            f.write(json.dumps(self.get_json(state1), indent=4, sort_keys=True))

        if self.iteration > 10 and abs(self.get_magnitude(state1.kinematics_estimated.position)) < 2.0:
            print('stopped')
            self.running = False

    def clean(self) -> None:
        for f in os.listdir('images'):
            os.remove(f)

        for f in os.listdir('segmentations'):
            os.remove(f)

        for f in os.listdir('states'):
            os.remove(f)

    def capture(self) -> None:
        self.running = True
        self.iteration: int = 0
        self.timestamps: Dict[int, str] = {}
        self.clean()

        while self.running:
            self.get_states()

            responses = self.client.simGetImages([
                airsim.ImageRequest("high_res", airsim.ImageType.Scene),
                airsim.ImageRequest("segment", airsim.ImageType.Segmentation)
            ], vehicle_name=self.observing_drone)
            print('Retrieved images: {}'.format(len(responses)))

            # do something with the images
            for response in responses:
                if response.pixels_as_float:
                    print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                    airsim.write_pfm(os.path.normpath('depth.pfm'), airsim.get_pfm_array(response))
                else:
                    print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                    image_type: str = 'images' if response.image_type == airsim.ImageType.Scene else 'segmentations'
                    airsim.write_file(os.path.normpath(f'{image_type}/image_{self.iteration}.png'), response.image_data_uint8)

            self.timestamps[self.iteration] = self.get_time_formatted()
            self.iteration += 1

    def land(self) -> None:
        print('landing...')
        self.align_north()

        f1 = self.client.landAsync(vehicle_name=self.observing_drone)
        f2 = self.client.landAsync(vehicle_name=self.target_drone)
        f1.join()
        f2.join()

    def finish(self) -> None:
        with open(f'states/timestamps.json', 'w') as f:
            f.write(json.dumps(self.timestamps, indent=4, sort_keys=True))


    def run(self) -> None:
        try:
            self.init()
            self.takeoff()
            self.fly_path()
            self.capture()
            self.land()
        finally:
            self.finish()


control = AirSimControl()
control.run()
