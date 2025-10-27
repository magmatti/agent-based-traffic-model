from dataclasses import dataclass

@dataclass
class Topology:
    lane_length: float = 200.0  # meters from stop line backward per approach
    speed_limit: float = 13.9   # ~50 km/h in m/s
    safe_gap: float = 5.0       # min bumper-to-bumper gap in meters
