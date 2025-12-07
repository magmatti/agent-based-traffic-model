from dataclasses import asdict

from traffic_sim.backends.base_backend import SimulationBackend
from traffic_sim.metrics.types import SimulationResult
from traffic_sim.metrics.timers import Timer


class OpenMPBackend(SimulationBackend):
    name = "openmp"

    def run(self) -> SimulationBackend:
        # TODO numba @njit(parallel=True) goes here
        with Timer() as t:
            total_sim_time = self.config.total_time

            vehicles_completed = 0
            avg_travel_time = 0.0
            avg_stops_per_vehicle = 0.0
            throughput = 0.0

        return SimulationResult(
            backend=self.name,
            config=asdict(self.config),
            wall_time_seconds=t.elapsed,
            total_simulated_time=total_sim_time,
            vehicles_completed=vehicles_completed,
            avg_travel_time=avg_travel_time,
            avg_stops_per_vehicle=avg_stops_per_vehicle,
            throughput_veh_per_min=throughput
        )
