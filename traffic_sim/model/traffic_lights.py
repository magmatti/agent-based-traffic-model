from dataclasses import dataclass
from typing import Dict

from .vehicles import Direction


@dataclass
class TrafficLightConfig:
    green_ns: float = 30.0
    green_ew: float = 30.0
    all_red: float = 2.0


class TrafficLightsController:
    """
    Traffic lights controller, NS/EW cycle.
    Returns dict {Direction: bool}, where True = greenLight
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
        Docstring for get_state
        
        :param self: Description
        :param t: Description
        :type t: float
        :return: Description
        :rtype: Dict[Direction, bool]
        """

        phase = t % self.cycle_duration

        green_ns = False
        green_ew = False

        if phase < self.config.green_ns:
            # phase 1 -> NS green, EW red
            green_ns = True
        elif phase < self.config.green_ns + self.config.all_red:
            # all red
            green_ns = False
            green_ew = False
        elif phase < self.config.green_ns + self.config.all_red + self.config.green_ew:
            # phase 2 -> EW green, NS red
            green_ew = True
        else:
            # all red
            green_ns = False
            green_ew = False

        state: Dict[Direction, bool] = {
            Direction.NORTH: green_ns,
            Direction.SOUTH: green_ns,
            Direction.EAST: green_ew,
            Direction.WEST: green_ew,
        }

        return state
