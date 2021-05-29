from __future__ import annotations
import airsim
from enum import Enum
import numpy as np

class Orientation(Enum):
    NORTH = 0,
    EAST  = 1,
    SOUTH = 2,
    WEST  = 3

    def __str__(self) -> str:
        return super().__str__().replace('Orientation.', '').lower()

    def get_heading(self) -> float:
        return {
            'NORTH': 0,
            'EAST':  90,
            'SOUTH': 180,
            'WEST':  270,
        }[str(self).upper()]

class Mode(Enum):
    ORBIT       = 0,
    COLLISION   = 1,

    def __str__(self) -> str:
        return super().__str__().replace('Mode.', '').lower()

class SimConfig:
    def __init__(self, base_name: str, height_name: str, center: airsim.Vector3r, orientation: Orientation,
                 radius: float, ground_height: float, orbit_speed: float, global_speed: airsim.Vector3r,
                 global_speed_name: str, mode: str, collision_angle: float) -> None:
        self.base_name: str = base_name
        self.height_name: str = height_name
        self.center: airsim.Vector3r = center
        self.orientation = orientation
        self.radius: float = radius
        self.ground_height: float = ground_height
        self.orbit_speed: float = orbit_speed
        self.global_speed: airsim.Vector3r = global_speed
        self.global_speed_name: str = global_speed_name
        self.mode: str = mode
        self.collision_angle: str = collision_angle

    @classmethod
    def get_mode(cls, mode_key: str) -> Mode:
        """Converts a str key to the Mode object.

        Args:
            mode_key (str): key that specifies the Mode to return

        Returns:
            Mode: The resulting orientation
        """
        options = [mode.name for mode in Mode]
        mode_key = mode_key.upper()

        if mode_key not in options:
            options_str = ', '.join(options)
            raise ValueError(
                f'Mode {mode_key} is not a valid orientation type, has to be one of {options_str}'
            )

        return Mode[mode_key]

    @classmethod
    def get_orientation(cls, orientation_key: str) -> Orientation:
        """Converts a str key to the Orientation object.

        Args:
            orientation_key (str): key that specifies the Orientation to return

        Returns:
            Orientation: The resulting orientation
        """
        options = [orientation.name for orientation in Orientation]
        orientation_key = orientation_key.upper()
        if orientation_key not in options:
            options_str = ', '.join(options)
            raise ValueError(
                f'Orientation {orientation_key} is not a valid orientation type, has to be one of {options_str}'
            )

        return Orientation[orientation_key]

    def __str__(self) -> str:
        return f'{self.base_name}-{self.mode}-{self.collision_angle}-{self.orientation}-{self.height_name}-{self.radius}-{self.orbit_speed}-{self.global_speed_name}'

    def is_different_location(self, other: SimConfig) -> bool:
        return self.base_name != other.base_name or self.mode == Mode.COLLISION

    def is_different_pose(self, other: SimConfig) -> bool:
        return self.orientation != other.orientation

    def is_different_height(self, other: SimConfig) -> bool:
        return self.height_name != other.height_name

    def is_different_simple(self, other: SimConfig) -> bool:
        return self.radius != other.radius or self.orbit_speed != other.orbit_speed or self.global_speed != other.global_speed

    def is_different(self, other: SimConfig) -> bool:
        return self.is_different_location(other) or self.is_different_pose(other) or self.is_different_height(other) or self.is_different_simple(other)

    def get_start_position(self, is_observer: bool) -> airsim.Vector3r:
        if self.mode == Mode.ORBIT:
            if is_observer:
                return self.center

            heading = np.deg2rad(self.orientation.get_heading() - 70)
            return self.center + airsim.Vector3r(np.cos(heading), np.sin(heading), 0.0) * self.radius
        elif self.mode == Mode.COLLISION:
            if is_observer:
                heading = np.deg2rad(self.orientation.get_heading() + 180)
            else:
                heading = np.deg2rad(self.orientation.get_heading() + self.collision_angle)

            return self.center + airsim.Vector3r(np.cos(heading), np.sin(heading), 0.0) * self.radius

