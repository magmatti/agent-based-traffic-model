from __future__ import annotations

import math
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


    def record_finished(self, v: Vehicle, now: float) -> None:
        if v.finish_time is None:
            return
        self.finished_count += 1
        self.finished_travel_times.append(v.finish_time - v.spawn_time)
        self.finished_stops.append(v.stops_count)


    def compute_summary(self, total_sim_time: float) -> Tuple[int, float, float, float]:
        if self.finished_count == 0:
            return 0, 0.0, 0.0, 0.0

        avg_travel = sum(self.finished_travel_times) / self.finished_count
        avg_stops = sum(self.finished_stops) / self.finished_count
        throughput_per_min = self.finished_count / (total_sim_time / 60.0)
        return self.finished_count, avg_travel, avg_stops, throughput_per_min
    

class WorldState:
    """
    Main world state: holds vehicles, road network and lights.
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        lights: TrafficLightsController,
        spawn_rate: float,
        max_vehicles: int,
        random_seed: int = 42,
        max_speed: float = 13.9,  # 13.9 [m/s] -> ~50 km/h
        safe_gap: float = 5.0, 
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


    # ---------- public API ----------


    def step(self, dt: float) -> None:
        """
        Each step of simulation:
        - lights update (with get_state() method),
        - spawning of new vehicles,
        - updating each vehicles directions,
        - deletion of finished cars
        """
        t_next = self.time + dt
        light_state = self.lights.get_state(self.time)

        # spawn
        self._spawn_vehicles(dt)

        # move update
        self._update_vehicles(dt, light_state)

        # deletion of finished + updating metrics
        self._remove_finished_and_update_metrics(t_next)

        self.time = t_next


    def get_metrics_summary(self, total_sim_time: float) -> Tuple[int, float, float, float]:
        return self.metrics_raw.compute_summary(total_sim_time)
    

    # ---------- internal logic ----------


    def _spawn_vehicles(self, dt: float) -> None:
        """
        For each direction, with a probability of 
        p = spawn_rate * dt, a new vehicle is added provided that max_vehicles has not been exceeded
        """
        if len(self.vehicles) >= self.max_vehicles:
            return

        spawn_prob = self.spawn_rate * dt
        for direction in Direction:
            if len(self.vehicles) >= self.max_vehicles:
                break

            if self.rng.random() < spawn_prob:
                v = self._create_vehicle(direction)
                self.vehicles.append(v)


    def _create_vehicle(self, direction: Direction) -> Vehicle:
        # choosing of driving direction (straight 60%, right 20%, left 20%)
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
        Movement update:
        - Process vehicles in each direction independently.
        """

        # group cars by direction
        by_dir: Dict[Direction, List[Vehicle]] = {d: [] for d in Direction}
        for v in self.vehicles:
            by_dir[v.direction].append(v)

        for d, vehicles in by_dir.items():
            if not vehicles:
                continue

            lane = self.road_network.get_lane(d)
            is_green = light_state[d]

            # isort from the furthest forward (largest postion)
            vehicles.sort(key=lambda v: v.position, reverse=True)

            # iterate from front to back to know the preceding vehicle
            front_vehicle: Vehicle | None = None
            for v in vehicles:
                # We assume a simple strategy: we drive at max_speed,
                # but we cannot:
                #  - hit the car in front of us,
                #  - pass the stop_line on a red light.
                desired_pos = v.position + v.max_speed * dt
                new_speed = v.max_speed

                # constraint imposed by the vehicle in front
                if front_vehicle is not None:
                    max_pos = front_vehicle.position - self.safe_gap
                    if desired_pos > max_pos:
                        # we must slow down / stop
                        desired_pos = max_pos
                        if desired_pos <= v.position + 1e-3:
                            new_speed = 0.0

                # constraint imposed by the red light
                if not is_green:
                    stop_pos = lane.stop_line_pos
                    if v.position < stop_pos and desired_pos >= stop_pos:
                        # we must stop before the stop line
                        # small margin
                        desired_pos = stop_pos - 0.5
                        if desired_pos <= v.position + 1e-3:
                            new_speed = 0.0

                # count stop (from >0 to 0)
                if v.speed > 0.1 and new_speed <= 0.1:
                    v.stops_count += 1

                v.position = max(0.0, min(desired_pos, lane.length + 10.0))
                v.speed = new_speed

                front_vehicle = v


    def _remove_finished_and_update_metrics(self, t_next: float) -> None:
        """
        deleting vehicles that crossed end of their line
        """
        remaining: List[Vehicle] = []
        for v in self.vehicles:
            lane = self.road_network.get_lane(v.direction)
            if v.position >= lane.length:
                v.mark_finished(t_next)
                self.metrics_raw.record_finished(v, t_next)
            else:
                remaining.append(v)
        self.vehicles = remaining
