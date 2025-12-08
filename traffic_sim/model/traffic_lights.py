from dataclasses import dataclass
from typing import Dict

from .vehicles import Direction


@dataclass
class TrafficLightConfig:
    green_ns: float = 30.0   # [s] green for N/S
    green_ew: float = 30.0   # [s] green for E/W
    all_red: float = 2.0     # [s] all-red phase between changes


class TrafficLightsController:
    """
    Simple 2-phase controller:
    - phase 1: NS green, EW red
    - phase 2: EW green, NS red
    Optional all-red phases between them.
    """

    def __init__(self, config: TrafficLightConfig | None = None) -> None:
        self.config = config or TrafficLightConfig()
        self.cycle_duration = (
            self.config.green_ns +
            self.config.all_red +
            self.config.green_ew +
            self.config.all_red
        )

    def get_state(self, t: float) -> Dict[Direction, bool]:
        """
        :param t: simulation time [s]
        :return: dict Direction -> True (green) / False (red)
        """
        phase = t % self.cycle_duration

        green_ns = False
        green_ew = False

        if phase < self.config.green_ns:
            # phase 1: NS green
            green_ns = True
        elif phase < self.config.green_ns + self.config.all_red:
            # all-red
            green_ns = False
            green_ew = False
        elif phase < self.config.green_ns + self.config.all_red + self.config.green_ew:
            # phase 2: EW green
            green_ew = True
        else:
            # all-red
            green_ns = False
            green_ew = False

        return {
            Direction.NORTH: green_ns,
            Direction.SOUTH: green_ns,
            Direction.EAST: green_ew,
            Direction.WEST: green_ew,
        }
