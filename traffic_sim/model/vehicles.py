from dataclasses import dataclass
from enum import IntEnum
from typing import Literal


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


# TurnChoice has to use one of the values in Literal
# thats why Literal type is used
TurnChoice = Literal["straight", "left", "right"]


@dataclass
class Vehicle: 
    id: int
    direction: Direction
    turn_choice = TurnChoice
    position: float
    speed: float
    max_speed: float
    spawn_time: float
    stops_count: int = 0
    finished: bool = False
    finihsed_time: float | None = None

    
    def mark_finished(self, t: float) -> None:
        self.finihsed = True
        self.finihsed_time = t
