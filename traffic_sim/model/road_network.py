from dataclasses import dataclass
from typing import Dict

from .vehicles import Direction


@dataclass(frozen=True)
class LaneGeometry:
    length: float                # lane length [m]
    stop_line_pos: float         # stop line position [m]
    intersection_start: float    # intersection start position [m]
    intersection_end: float      # intersection end position [m]


class RoadNetwork:
    """
    Simple 1D road network:
    one lane per direction, vehicles travel from position=0 to position=length.
    """

    def __init__(
        self,
        lane_length: float = 100.0,
        stop_line_from_center: float = 5.0,
        intersection_width: float = 10.0,
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

        # same geometry for all directions
        self.lanes: Dict[Direction, LaneGeometry] = {
            d: geom for d in Direction
        }

    def get_lane(self, direction: Direction) -> LaneGeometry:
        """Return lane geometry for the given direction."""
        return self.lanes[direction]
