from dataclasses import dataclass
from typing import Dict

from .vehicles import Direction


# (frozen=True) -> immutable dataclass 
@dataclass(frozen=True)
class LaneGeometry:
    length: float
    stop_line_pos: float
    intersection_start: float
    intersection_end: float


class RoadNetwork:
    """
    Simple 1D model: one lane in each direction.
    Vehicle goes from position=0 to position=length, than leaves the map.
    """

    def __init__(
            self,
            lane_length: float = 100.0,
            stop_line_from_center: float = 5.0,
            intersection_width: float = 10.0
    ) -> None:
        self.lane_length = lane_length
        center = lane_length / 2.0
        stop_line_pos = center - stop_line_from_center
        intersection_start = center - intersection_width / 2.0
        intersection_end = center + intersection_width / 2.0

        geom = LaneGeometry(
            length=lane_length,
            stop_line_pos=stop_line_pos,
            intersection_start=intersection_start,
            intersection_end=intersection_end,
        )

        # identical geometry for all directions
        self.lanes: Dict[Direction, LaneGeometry] = {
            d: geom for d in Direction
        }

        def get_lane(self, direction: Direction) -> LaneGeometry:
            return self.lanes[direction]
