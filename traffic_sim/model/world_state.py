from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable

import numpy as np
from numba import njit, prange, cuda

from .road_network import RoadNetwork
from .traffic_lights import TrafficLightsController
from .vehicles import Vehicle, Direction, TurnChoice


@dataclass
class SimulationMetricsRaw:
    finished_travel_times: List[float] = field(default_factory=list)
    finished_stops: List[int] = field(default_factory=list)
    finished_count: int = 0

    total_spawned: int = 0  # number of vehicles spawned in total

    def record_spawned(self, n: int = 1) -> None:
        """Increment the count of spawned vehicles."""
        self.total_spawned += n

    def record_finished(self, v: Vehicle) -> None:
        """Store metrics for a vehicle that finished its trip."""
        if v.finish_time is None:
            return
        self.finished_count += 1
        self.finished_travel_times.append(v.finish_time - v.spawn_time)
        self.finished_stops.append(v.stops_count)

    def compute_summary(self, total_sim_time: float) -> Tuple[int, float, float, float]:
        """
        Compute derived statistics:
        - total completed vehicles
        - average travel time
        - average stops per vehicle
        - throughput (vehicles per minute)
        """
        if self.finished_count == 0:
            return 0, 0.0, 0.0, 0.0

        avg_travel = sum(self.finished_travel_times) / self.finished_count
        avg_stops = sum(self.finished_stops) / self.finished_count
        throughput_per_min = self.finished_count / (total_sim_time / 60.0)

        return self.finished_count, avg_travel, avg_stops, throughput_per_min


@njit(parallel=True)
def update_lanes_kernel(
    positions: np.ndarray,
    speeds: np.ndarray,
    stops: np.ndarray,
    counts: np.ndarray,
    is_green_arr: np.ndarray,
    stop_line_arr: np.ndarray,
    lane_length_arr: np.ndarray,
    safe_gap: float,
    dt: float,
    max_speed: float,
) -> None:
    """
    Numba-parallel kernel that updates all lanes.

    positions, speeds, stops have shape (num_lanes, max_n_per_lane)
    counts[lane_idx] says how many vehicles are valid in that lane.
    """
    num_lanes = counts.shape[0]

    for lane_idx in prange(num_lanes):
        n = counts[lane_idx]
        if n == 0:
            continue

        is_green = is_green_arr[lane_idx]
        stop_line = stop_line_arr[lane_idx]
        lane_length = lane_length_arr[lane_idx]

        for i in range(n):
            old_pos = positions[lane_idx, i]
            old_speed = speeds[lane_idx, i]

            desired_pos = old_pos + max_speed * dt
            new_speed = max_speed

            # Collision avoidance: keep safe gap to the vehicle in front
            if i > 0:
                front_pos = positions[lane_idx, i - 1]
                max_pos = front_pos - safe_gap
                if desired_pos > max_pos:
                    desired_pos = max_pos
                    if desired_pos <= old_pos + 1e-3:
                        new_speed = 0.0

            # Respect red light: stop before stop line
            if not is_green:
                if old_pos < stop_line and desired_pos >= stop_line:
                    desired_pos = stop_line - 0.5
                    if desired_pos <= old_pos + 1e-3:
                        new_speed = 0.0

            # Count stop events (speed > 0 -> 0)
            if old_speed > 0.1 and new_speed <= 0.1:
                stops[lane_idx, i] += 1

            # Clamp position to reasonable bounds
            if desired_pos < 0.0:
                desired_pos = 0.0
            if desired_pos > lane_length + 20.0:
                desired_pos = lane_length + 20.0

            positions[lane_idx, i] = desired_pos
            speeds[lane_idx, i] = new_speed


@cuda.jit
def update_lanes_kernel_cuda(
    old_positions,   # float64[:, :]
    old_speeds,      # float64[:, :]
    stops,           # int32[:, :]
    counts,          # int32[:]
    is_green_arr,    # bool_[:]
    stop_line_arr,   # float64[:]
    lane_length_arr, # float64[:]
    safe_gap: float,
    dt: float,
    max_speed: float,
):
    """
    CUDA kernel that updates all lanes in parallel.

    Grid:
        blockIdx.x -> lane index
        threadIdx.x -> vehicle index in that lane
    """
    lane_idx = cuda.blockIdx.x
    i = cuda.threadIdx.x

    num_lanes = counts.shape[0]
    if lane_idx >= num_lanes:
        return

    n = counts[lane_idx]
    if i >= n:
        return

    is_green = is_green_arr[lane_idx]
    stop_line = stop_line_arr[lane_idx]
    lane_length = lane_length_arr[lane_idx]

    old_pos = old_positions[lane_idx, i]
    old_speed = old_speeds[lane_idx, i]

    desired_pos = old_pos + max_speed * dt
    new_speed = max_speed

    # Collision avoidance: use old_positions to avoid data races
    if i > 0:
        front_pos = old_positions[lane_idx, i - 1]
        max_pos = front_pos - safe_gap
        if desired_pos > max_pos:
            desired_pos = max_pos
            if desired_pos <= old_pos + 1e-3:
                new_speed = 0.0

    # Respect red light: stop before stop line
    if not is_green:
        if old_pos < stop_line and desired_pos >= stop_line:
            desired_pos = stop_line - 0.5
            if desired_pos <= old_pos + 1e-3:
                new_speed = 0.0

    # Count stop events (speed > 0 -> 0)
    if old_speed > 0.1 and new_speed <= 0.1:
        stops[lane_idx, i] += 1

    # Clamp position to reasonable bounds
    if desired_pos < 0.0:
        desired_pos = 0.0
    if desired_pos > lane_length + 20.0:
        desired_pos = lane_length + 20.0

    # Write results back to old_positions / old_speeds arrays (they will be copied back)
    old_positions[lane_idx, i] = desired_pos
    old_speeds[lane_idx, i] = new_speed


class WorldState:
    """
    Manages the full state of the traffic simulation:
    - vehicles
    - road/lane geometry
    - traffic lights
    - spawning
    - movement logic (sequential, OpenMP-like)
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        lights: TrafficLightsController,
        spawn_rate: float,
        max_vehicles: int,
        random_seed: int = 42,
        max_speed: float = 13.9,  # ~50 km/h
        safe_gap: float = 5.0,    # minimum distance between vehicles
        active_directions: Iterable[Direction] | None = None,
    ) -> None:

        self.road_network = road_network
        self.lights = lights
        self.spawn_rate = spawn_rate
        self.max_vehicles = max_vehicles
        self.max_speed = max_speed
        self.safe_gap = safe_gap

        # Which directions are actually simulated in this world (for MPI domain decomposition)
        if active_directions is None:
            self.active_directions: List[Direction] = [d for d in Direction]
        else:
            self.active_directions = list(active_directions)

        # Fixed order of directions used by the OpenMP-like kernel
        self.directions_order: List[Direction] = list(self.active_directions)

        self.time: float = 0.0
        self._next_vehicle_id: int = 0
        self.vehicles: List[Vehicle] = []

        self.metrics_raw = SimulationMetricsRaw()

        self.rng = random.Random(random_seed)

    # ------------------------ PUBLIC API ------------------------

    def step(self, dt: float) -> None:
        """
        Sequential step:
        1) spawn new vehicles
        2) update movement of all vehicles in pure Python
        3) remove finished vehicles + collect metrics
        """
        t_next = self.time + dt
        light_state = self.lights.get_state(self.time)

        self._spawn_vehicles(dt)
        self._update_vehicles_sequential(dt, light_state)
        self._remove_finished_and_update_metrics(t_next)

        self.time = t_next

    def step_openmp(self, dt: float) -> None:
        """
        OpenMP-like step:
        1) spawn new vehicles
        2) update movement using a Numba-parallel kernel
        3) remove finished vehicles + collect metrics
        """
        t_next = self.time + dt
        light_state = self.lights.get_state(self.time)

        self._spawn_vehicles(dt)
        self._update_vehicles_openmp(dt, light_state)
        self._remove_finished_and_update_metrics(t_next)

        self.time = t_next

    def step_cuda(self, dt: float) -> None:
        """
        CUDA-like step:
        1) spawn new vehicles
        2) update movement using a Numba CUDA kernel
        3) remove finished vehicles + collect metrics

        NOTE: this requires a CUDA-capable GPU and proper driver setup.
        """
        t_next = self.time + dt
        light_state = self.lights.get_state(self.time)

        self._spawn_vehicles(dt)
        self._update_vehicles_cuda(dt, light_state)
        self._remove_finished_and_update_metrics(t_next)

        self.time = t_next


    def get_metrics_summary(self, total_sim_time: float) -> Tuple[int, float, float, float]:
        """Return the aggregated simulation metrics."""
        return self.metrics_raw.compute_summary(total_sim_time)


    def get_debug_stats(self) -> Dict[str, int]:
        """
        Return simple debug stats:
        - total spawned vehicles
        - how many vehicles are still in the world
        """
        return {
            "total_spawned": self.metrics_raw.total_spawned,
            "vehicles_in_world_end": len(self.vehicles),
        }


    # ------------------------ INTERNAL LOGIC ------------------------

    def _spawn_vehicles(self, dt: float) -> None:
        """
        Spawn new vehicles based on spawn_rate probability.
        For each active direction:
        - probability = spawn_rate * dt
        - cap at max_vehicles
        """
        if len(self.vehicles) >= self.max_vehicles:
            return

        spawn_prob = self.spawn_rate * dt
        spawned_now = 0

        for direction in self.active_directions:
            if len(self.vehicles) >= self.max_vehicles:
                break

            if self.rng.random() < spawn_prob:
                v = self._create_vehicle(direction)
                self.vehicles.append(v)
                spawned_now += 1

        if spawned_now > 0:
            self.metrics_raw.record_spawned(spawned_now)


    def _create_vehicle(self, direction: Direction) -> Vehicle:
        """
        Create a new vehicle entering from the given direction.
        Turning behavior is randomly chosen.
        """
        r = self.rng.random()
        if r < 0.6:
            turn: TurnChoice = "straight"
        elif r < 0.8:
            turn = "right"
        else:
            turn = "left"

        vid = self._next_vehicle_id
        self._next_vehicle_id += 1

        return Vehicle(
            id=vid,
            direction=direction,
            turn_choice=turn,
            position=0.0,
            speed=self.max_speed,
            max_speed=self.max_speed,
            spawn_time=self.time,
        )


    # ---------- Sequential update (used by SequentialBackend and MPI ranks) ----------


    def _update_vehicles_sequential(self, dt: float, light_state: Dict[Direction, bool]) -> None:
        """
        Pure Python sequential update of vehicle movement.
        This is the reference implementation.
        """
        by_dir: Dict[Direction, List[Vehicle]] = {d: [] for d in self.active_directions}
        for v in self.vehicles:
            by_dir[v.direction].append(v)

        for d in self.active_directions:
            vehicles = by_dir[d]
            if not vehicles:
                continue

            lane = self.road_network.get_lane(d)
            is_green = light_state[d]

            vehicles.sort(key=lambda v: v.position, reverse=True)

            front_vehicle: Vehicle | None = None
            for v in vehicles:
                desired_pos = v.position + v.max_speed * dt
                new_speed = v.max_speed

                if front_vehicle is not None:
                    max_pos = front_vehicle.position - self.safe_gap
                    if desired_pos > max_pos:
                        desired_pos = max_pos
                        if desired_pos <= v.position + 1e-3:
                            new_speed = 0.0

                if not is_green:
                    stop_line = lane.stop_line_pos
                    if v.position < stop_line and desired_pos >= stop_line:
                        desired_pos = stop_line - 0.5
                        if desired_pos <= v.position + 1e-3:
                            new_speed = 0.0

                if v.speed > 0.1 and new_speed <= 0.1:
                    v.stops_count += 1

                if desired_pos < 0.0:
                    desired_pos = 0.0
                if desired_pos > lane.length + 20.0:
                    desired_pos = lane.length + 20.0

                v.position = desired_pos
                v.speed = new_speed

                front_vehicle = v


    # ---------- OpenMP-like update using Numba ----------


    def _update_vehicles_openmp(self, dt: float, light_state: Dict[Direction, bool]) -> None:
        """
        Update vehicle movement using a Numba-parallel kernel.
        The logic is equivalent to the sequential version, but operates
        on NumPy arrays for better performance and easier parallelization.
        """
        directions_order = self.directions_order
        num_lanes = len(directions_order)

        # Group vehicles by direction and sort by position descending
        vehicles_per_lane: List[List[Vehicle]] = []
        max_n = 0

        for d in directions_order:
            lane_vehicles = [v for v in self.vehicles if v.direction == d]
            lane_vehicles.sort(key=lambda v: v.position, reverse=True)
            vehicles_per_lane.append(lane_vehicles)
            if len(lane_vehicles) > max_n:
                max_n = len(lane_vehicles)

        if max_n == 0:
            return

        # Allocate arrays for positions, speeds, stops
        positions = np.zeros((num_lanes, max_n), dtype=np.float64)
        speeds = np.zeros((num_lanes, max_n), dtype=np.float64)
        stops = np.zeros((num_lanes, max_n), dtype=np.int32)
        counts = np.zeros(num_lanes, dtype=np.int32)

        is_green_arr = np.zeros(num_lanes, dtype=np.bool_)
        stop_line_arr = np.zeros(num_lanes, dtype=np.float64)
        lane_length_arr = np.zeros(num_lanes, dtype=np.float64)

        # Fill arrays from current vehicle objects
        for lane_idx, d in enumerate(directions_order):
            lane_vehicles = vehicles_per_lane[lane_idx]
            n = len(lane_vehicles)
            counts[lane_idx] = n

            lane = self.road_network.get_lane(d)
            is_green_arr[lane_idx] = light_state[d]
            stop_line_arr[lane_idx] = lane.stop_line_pos
            lane_length_arr[lane_idx] = lane.length

            for i, v in enumerate(lane_vehicles):
                positions[lane_idx, i] = v.position
                speeds[lane_idx, i] = v.speed
                stops[lane_idx, i] = v.stops_count

        # Run Numba kernel
        update_lanes_kernel(
            positions,
            speeds,
            stops,
            counts,
            is_green_arr,
            stop_line_arr,
            lane_length_arr,
            self.safe_gap,
            dt,
            self.max_speed,
        )

        # Write back updated values to vehicle objects
        for lane_idx, d in enumerate(directions_order):
            lane_vehicles = vehicles_per_lane[lane_idx]
            n = counts[lane_idx]
            for i in range(n):
                v = lane_vehicles[i]
                v.position = float(positions[lane_idx, i])
                v.speed = float(speeds[lane_idx, i])
                v.stops_count = int(stops[lane_idx, i])


    def _update_vehicles_cuda(self, dt: float, light_state: Dict[Direction, bool]) -> None:
        """
        Update vehicle movement using a Numba CUDA kernel.

        This is similar to the OpenMP-like implementation, but the core
        update loop runs on the GPU.
        """
        directions_order = self.directions_order
        num_lanes = len(directions_order)

        # Group vehicles by direction and sort by position descending
        vehicles_per_lane: List[List[Vehicle]] = []
        max_n = 0

        for d in directions_order:
            lane_vehicles = [v for v in self.vehicles if v.direction == d]
            lane_vehicles.sort(key=lambda v: v.position, reverse=True)
            vehicles_per_lane.append(lane_vehicles)
            if len(lane_vehicles) > max_n:
                max_n = len(lane_vehicles)

        if max_n == 0:
            return

        # Allocate host arrays
        old_positions = np.zeros((num_lanes, max_n), dtype=np.float64)
        old_speeds = np.zeros((num_lanes, max_n), dtype=np.float64)
        stops = np.zeros((num_lanes, max_n), dtype=np.int32)
        counts = np.zeros(num_lanes, dtype=np.int32)

        is_green_arr = np.zeros(num_lanes, dtype=np.bool_)
        stop_line_arr = np.zeros(num_lanes, dtype=np.float64)
        lane_length_arr = np.zeros(num_lanes, dtype=np.float64)

        # Fill arrays from vehicles and lane geometry
        for lane_idx, d in enumerate(directions_order):
            lane_vehicles = vehicles_per_lane[lane_idx]
            n = len(lane_vehicles)
            counts[lane_idx] = n

            lane = self.road_network.get_lane(d)
            is_green_arr[lane_idx] = light_state[d]
            stop_line_arr[lane_idx] = lane.stop_line_pos
            lane_length_arr[lane_idx] = lane.length

            for i, v in enumerate(lane_vehicles):
                old_positions[lane_idx, i] = v.position
                old_speeds[lane_idx, i] = v.speed
                stops[lane_idx, i] = v.stops_count

        # Transfer data to GPU
        d_old_positions = cuda.to_device(old_positions)
        d_old_speeds = cuda.to_device(old_speeds)
        d_stops = cuda.to_device(stops)
        d_counts = cuda.to_device(counts)
        d_is_green = cuda.to_device(is_green_arr)
        d_stop_line = cuda.to_device(stop_line_arr)
        d_lane_length = cuda.to_device(lane_length_arr)

        # Configure CUDA grid: one block per lane, max_n threads per block (clamped)
        threads_per_block = min(256, max_n)
        blocks_per_grid = num_lanes

        update_lanes_kernel_cuda[blocks_per_grid, threads_per_block](
            d_old_positions,
            d_old_speeds,
            d_stops,
            d_counts,
            d_is_green,
            d_stop_line,
            d_lane_length,
            self.safe_gap,
            dt,
            self.max_speed,
        )

        # Copy results back to host
        d_old_positions.copy_to_host(old_positions)
        d_old_speeds.copy_to_host(old_speeds)
        d_stops.copy_to_host(stops)

        # Write back to vehicle objects
        for lane_idx, d in enumerate(directions_order):
            lane_vehicles = vehicles_per_lane[lane_idx]
            n = counts[lane_idx]
            for i in range(n):
                v = lane_vehicles[i]
                v.position = float(old_positions[lane_idx, i])
                v.speed = float(old_speeds[lane_idx, i])
                v.stops_count = int(stops[lane_idx, i])


    # ---------- Finish handling ----------

    def _remove_finished_and_update_metrics(self, t_next: float) -> None:
        """
        Remove vehicles that have reached the end of their lane.
        A vehicle is considered finished if: position >= lane.length.
        """
        remaining: List[Vehicle] = []

        for v in self.vehicles:
            lane = self.road_network.get_lane(v.direction)
            if v.position >= lane.length:
                v.mark_finished(t_next)
                self.metrics_raw.record_finished(v)
            else:
                remaining.append(v)

        self.vehicles = remaining
