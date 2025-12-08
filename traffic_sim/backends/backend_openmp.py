from dataclasses import asdict

from numba import set_num_threads

from traffic_sim.backends.base_backend import SimulationBackend
from traffic_sim.config import SimulationConfig
from traffic_sim.metrics.types import SimulationResult
from traffic_sim.metrics.timers import Timer
from traffic_sim.model.road_network import RoadNetwork
from traffic_sim.model.traffic_lights import TrafficLightsController, TrafficLightConfig
from traffic_sim.model.world_state import WorldState


class OpenMPBackend(SimulationBackend):
    """
    OpenMP-like backend using Numba's parallel CPU execution.
    It uses the same WorldState model, but calls step_openmp()
    which updates vehicle movement via a Numba @njit(parallel=True) kernel.
    """

    name = "openmp"

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        # Configure the number of threads used by Numba
        if self.config.num_threads > 0:
            set_num_threads(self.config.num_threads)

        self.road_network = RoadNetwork(
            lane_length=100.0,
            stop_line_from_center=5.0,
            intersection_width=10.0,
        )

        lights_cfg = TrafficLightConfig(
            green_ns=30.0,
            green_ew=30.0,
            all_red=2.0,
        )
        self.lights = TrafficLightsController(lights_cfg)

        self.world = WorldState(
            road_network=self.road_network,
            lights=self.lights,
            spawn_rate=self.config.spawn_rate,
            max_vehicles=self.config.max_vehicles,
            random_seed=self.config.random_seed,
            max_speed=13.9,
            safe_gap=5.0,
        )


    def run(self) -> SimulationResult:
        cfg: SimulationConfig = self.config
        total_time = cfg.total_time
        dt = cfg.dt

        steps = int(total_time / dt)

        # Warm-up step to trigger Numba JIT compilation (not measured)
        if steps > 0:
            self.world.step_openmp(dt)

        with Timer() as t:
            for _ in range(steps - 1):
                self.world.step_openmp(dt)

        vehicles_completed, avg_travel, avg_stops, throughput = \
            self.world.get_metrics_summary(total_time)

        debug_stats = self.world.get_debug_stats()

        return SimulationResult(
            backend=self.name,
            config=asdict(cfg),
            wall_time_seconds=t.elapsed,
            total_simulated_time=total_time,
            vehicles_completed=vehicles_completed,
            avg_travel_time=avg_travel,
            avg_stops_per_vehicle=avg_stops,
            throughput_veh_per_min=throughput,
            extra_stats=debug_stats,
        )

