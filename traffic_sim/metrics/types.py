from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SimulationResult:
    backend: str
    config: Dict[str, Any]

    # total time 
    wall_time_seconds: float
    total_simulated_time: float

    # statystyki ruchu
    vehicles_completed: int
    avg_travel_time: float
    avg_stops_per_vehicle: float
    # [veh/min]
    throughput_veh_per_min: float

    # cokolwiek dodatkowego (np. rozkłady, max kolejka itp.)
    extra_stats: Dict[str, Any] = field(default_factory=dict)
