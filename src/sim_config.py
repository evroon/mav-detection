import airsim
from enum import Enum

class Orientation(Enum):
    NORTH = 0,
    EAST  = 1,
    SOUTH = 2,
    WEST  = 3

    def __str__(self) -> str:
        return super().__str__().replace('Orientation.', '')

    def get_heading(self) -> float:
        return {
            'NORTH': 0,
            'EAST':  90,
            'SOUTH': 180,
            'WEST':  270,
        }[str(self)]

class SimConfig:
    def __init__(self, basename: str, center: airsim.Vector3r, orientation: Orientation, radius: float) -> None:
        self.basename: str = basename
        self.center: airsim.Vector3r = airsim.Vector3r()
        self.orientation = Orientation.NORTH
        self.radius: float = radius


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
        return f'{self.basename}-{self.orientation}-{self.center.z_val*10:00d}'
