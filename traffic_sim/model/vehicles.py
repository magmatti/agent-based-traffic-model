from dataclasses import dataclass
from enum import IntEnum
from typing import Literal


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


TurnChoice = Literal["straight", "left", "right"]


@dataclass
class Vehicle:
    id: int
    direction: Direction
    turn_choice: TurnChoice      # turning decision
    position: float              # [m] from the start of the lane
    speed: float                 # [m/s]
    max_speed: float             # [m/s]
    spawn_time: float            # time when the vehicle entered the world

    stops_count: int = 0         # how many times the vehicle had to stop
    finished: bool = False
    finish_time: float | None = None

    def mark_finished(self, t: float) -> None:
        self.finished = True
        self.finish_time = t
