from dataclasses import dataclass, field
import numpy as np

@dataclass
class Metrics:
    departures: int = 0
    total_travel_time: float = 0.0
    total_stops: int = 0
    max_queue: dict = field(default_factory=lambda: {d:0 for d in 'NSEW'})

    def record_departure(self, travel_time: float, stops: int):
        self.departures += 1
        self.total_travel_time += travel_time
        self.total_stops += stops

    def update_queue(self, queues):
        for d in 'NSEW':
            self.max_queue[d] = max(self.max_queue[d], queues.get(d,0))

    def summary(self, sim_minutes: float):
        if self.departures == 0:
            avg_t = 0.0
            avg_s = 0.0
        else:
            avg_t = self.total_travel_time / self.departures
            avg_s = self.total_stops / self.departures
        throughput = self.departures / sim_minutes if sim_minutes > 0 else 0.0
        return {
            'departures': self.departures,
            'throughput_veh_per_min': throughput,
            'avg_travel_time_s': avg_t,
            'avg_stops_per_vehicle': avg_s,
            'max_queue_per_approach': self.max_queue
        }
