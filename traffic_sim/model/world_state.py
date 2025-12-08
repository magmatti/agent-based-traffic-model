from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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


class WorldState:
    """
    Manages the full state of the traffic simulation:
    - vehicles
    - road/lane geometry
    - traffic lights
    - spawning
    - movement logic
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
    ) -> None:

        self.road_network = road_network
        self.lights = lights
        self.spawn_rate = spawn_rate
        self.max_vehicles = max_vehicles
        self.max_speed = max_speed
        self.safe_gap = safe_gap

        self.time: float = 0.0
        self._next_vehicle_id: int = 0
        self.vehicles: List[Vehicle] = []

        self.metrics_raw = SimulationMetricsRaw()

        self.rng = random.Random(random_seed)

    # ------------------------ PUBLIC API ------------------------

    def step(self, dt: float) -> None:
        """
        Execute one simulation step:
        1) spawn new vehicles
        2) update movement of all vehicles
        3) remove finished vehicles + collect metrics
        """
        t_next = self.time + dt

        light_state = self.lights.get_state(self.time)

        # Step 1: spawning
        self._spawn_vehicles(dt)

        # Step 2: movement update
        self._update_vehicles(dt, light_state)

        # Step 3: remove completed vehicles
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
        For each direction:
        - probability = spawn_rate * dt
        - cap at max_vehicles
        """
        if len(self.vehicles) >= self.max_vehicles:
            return

        spawn_prob = self.spawn_rate * dt
        spawned_now = 0

        for direction in Direction:
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

    def _update_vehicles(self, dt: float, light_state: Dict[Direction, bool]) -> None:
        """
        Update movement for each direction independently:
        - sort vehicles by position (frontmost first)
        - for each vehicle:
            * try to move with max_speed
            * adjust to avoid collision with the front vehicle
            * stop before red light stop line
            * update stop counter
        """
        # Group vehicles by direction
        by_dir: Dict[Direction, List[Vehicle]] = {d: [] for d in Direction}
        for v in self.vehicles:
            by_dir[v.direction].append(v)

        for d, vehicles in by_dir.items():
            if not vehicles:
                continue

            lane = self.road_network.get_lane(d)
            is_green = light_state[d]

            # Sort by descending position (frontmost vehicle first)
            vehicles.sort(key=lambda v: v.position, reverse=True)

            front_vehicle: Vehicle | None = None
            for v in vehicles:

                desired_pos = v.position + v.max_speed * dt
                new_speed = v.max_speed

                # Collision avoidance
                if front_vehicle is not None:
                    max_pos = front_vehicle.position - self.safe_gap
                    if desired_pos > max_pos:
                        desired_pos = max_pos
                        if desired_pos <= v.position + 1e-3:
                            new_speed = 0.0

                # Stop at red light
                if not is_green:
                    stop_line = lane.stop_line_pos
                    if v.position < stop_line and desired_pos >= stop_line:
                        desired_pos = stop_line - 0.5
                        if desired_pos <= v.position + 1e-3:
                            new_speed = 0.0

                # Count stop event
                if v.speed > 0.1 and new_speed <= 0.1:
                    v.stops_count += 1

                # Clamp to safe bounds
                if desired_pos < 0.0:
                    desired_pos = 0.0
                if desired_pos > lane.length + 20.0:
                    desired_pos = lane.length + 20.0

                v.position = desired_pos
                v.speed = new_speed

                front_vehicle = v

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
