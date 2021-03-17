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

class SimConfig:
    def __init__(self, base_name: str, height_name: str, center: airsim.Vector3r, orientation: Orientation, radius: float, ground_height: float) -> None:
        self.base_name: str = base_name
        self.height_name: str = height_name
        self.center: airsim.Vector3r = center
        self.orientation = orientation
        self.radius: float = radius
        self.ground_height = ground_height


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
                f'Mode {orientation_key} is not a valid orientation type, has to be one of {options_str}'
            )

        return Orientation[orientation_key]

    def __str__(self) -> str:
        return f'{self.base_name}-{self.orientation}-{self.height_name}-{self.radius}'

    def full_name(self) -> str:
        return f'{self.base_name}-{self.orientation}-{self.height_name} ({self.radius}m)'

    def is_different_location(self, other: SimConfig) -> bool:
        return self.base_name != other.base_name

    def is_different_pose(self, other: SimConfig) -> bool:
        return self.orientation != other.orientation

    def is_different_height(self, other: SimConfig) -> bool:
        return self.height_name != other.height_name

    def is_different_radius(self, other: SimConfig) -> bool:
        return self.radius != other.radius

    def is_different(self, other: SimConfig) -> bool:
        return self.is_different_location(other) or self.is_different_pose(other) or self.is_different_height(other) or self.is_different_radius(other)

    def get_start_position(self, is_observer: bool) -> airsim.Vector3r:
        if is_observer:
            return self.center

        heading = self.orientation.get_heading() / 180.0 * np.pi
        return self.center + airsim.Vector3r(-np.cos(heading) * self.radius, -np.sin(heading) * self.radius, 0.0)
