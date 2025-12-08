from dataclasses import asdict

from mpi4py import MPI

from traffic_sim.backends.base_backend import SimulationBackend
from traffic_sim.config import SimulationConfig
from traffic_sim.metrics.types import SimulationResult
from traffic_sim.metrics.timers import Timer
from traffic_sim.model.road_network import RoadNetwork
from traffic_sim.model.traffic_lights import TrafficLightsController, TrafficLightConfig
from traffic_sim.model.world_state import WorldState
from traffic_sim.model.vehicles import Direction


class MPIBackend(SimulationBackend):
    """
    MPI backend using domain decomposition by directions.

    Each MPI rank simulates only a subset of directions, for example:
    - rank 0: NORTH
    - rank 1: EAST
    - rank 2: SOUTH
    - rank 3: WEST

    Ranks compute local metrics, which are then reduced (summed) to rank 0.
    Global averages and throughput are computed on rank 0 and broadcast
    back to all ranks.
    """

    name = "mpi"

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Assign directions to ranks in a simple round-robin fashion
        all_dirs = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        active_dirs = [d for i, d in enumerate(all_dirs) if i % max(self.size, 1) == self.rank]

        # If there are more ranks than directions, some ranks may get no directions
        # They will effectively simulate an empty world (no vehicles).
        self.active_directions = active_dirs

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
            random_seed=self.config.random_seed + self.rank,  # different seed per rank
            max_speed=13.9,
            safe_gap=5.0,
            active_directions=self.active_directions,
        )

    def run(self) -> SimulationResult:
        cfg: SimulationConfig = self.config
        total_time = cfg.total_time
        dt = cfg.dt
        steps = int(total_time / dt)

        # Sequential update per rank; domain decomposition is across ranks.
        with Timer() as t:
            for _ in range(steps):
                self.world.step(dt)

        # Local metrics (per rank)
        raw = self.world.metrics_raw
        local_finished = raw.finished_count
        local_sum_travel = float(sum(raw.finished_travel_times))
        local_sum_stops = float(sum(raw.finished_stops))
        local_spawned = raw.total_spawned

        # Local wall time
        local_wall = t.elapsed

        # Reduce metrics to rank 0
        comm = self.comm
        global_finished = comm.reduce(local_finished, op=MPI.SUM, root=0)
        global_sum_travel = comm.reduce(local_sum_travel, op=MPI.SUM, root=0)
        global_sum_stops = comm.reduce(local_sum_stops, op=MPI.SUM, root=0)
        global_spawned = comm.reduce(local_spawned, op=MPI.SUM, root=0)
        global_wall = comm.reduce(local_wall, op=MPI.MAX, root=0)  # max wall time across ranks

        if self.rank == 0 and global_finished > 0:
            avg_travel = global_sum_travel / global_finished
            avg_stops = global_sum_stops / global_finished
            throughput = global_finished / (total_time / 60.0)
        else:
            avg_travel = 0.0
            avg_stops = 0.0
            throughput = 0.0

        # Broadcast global metrics and wall time to all ranks
        global_data = (global_finished, avg_travel, avg_stops, throughput,
                       global_spawned, global_wall)
        global_data = comm.bcast(global_data, root=0)

        vehicles_completed, avg_travel, avg_stops, throughput, global_spawned, wall_time = global_data

        debug_stats = {
            "num_ranks": self.size,
            "rank": self.rank,
            "local_spawned": local_spawned,
            "local_finished": local_finished,
            "global_spawned": global_spawned,
        }

        return SimulationResult(
            backend=self.name,
            config=asdict(cfg),
            wall_time_seconds=wall_time,
            total_simulated_time=total_time,
            vehicles_completed=vehicles_completed,
            avg_travel_time=avg_travel,
            avg_stops_per_vehicle=avg_stops,
            throughput_veh_per_min=throughput,
            extra_stats=debug_stats,
        )
