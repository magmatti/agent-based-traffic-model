from dataclasses import dataclass
import numpy as np

DIRECTIONS = ['N','S','E','W']

@dataclass
class Fleet:
    # Per-direction arrays: positions counted from stop line backward (0 at stop line)
    x: dict
    v: dict
    stops: dict
    at_stopline: dict
    active: dict

    @staticmethod
    def empty(max_per_dir: int = 500):
        x = {d: np.empty(max_per_dir, dtype=np.float32) for d in DIRECTIONS}
        v = {d: np.empty(max_per_dir, dtype=np.float32) for d in DIRECTIONS}
        stops = {d: np.zeros(max_per_dir, dtype=np.int32) for d in DIRECTIONS}
        at_stopline = {d: np.zeros(max_per_dir, dtype=np.bool_) for d in DIRECTIONS}
        birth_time = {d: np.zeros(max_per_dir, dtype=np.float32) for d in DIRECTIONS}
        active = {d: 0 for d in DIRECTIONS}
        fleet = Fleet(x,v,stops,at_stopline,active)
        fleet.birth_time = birth_time
        return fleet

    def spawn(self, d: str, lane_len: float, v0: float = 0.0, t: float = 0.0):
        i = self.active[d]
        self.x[d][i] = lane_len
        self.v[d][i] = v0
        self.stops[d][i] = 0
        self.at_stopline[d][i] = False
        self.birth_time[d][i] = t
        self.active[d] += 1

    def headways(self, d: str):
        n = self.active[d]
        if n <= 1:
            out = np.full(n, 1e9, dtype=np.float32)
        else:
            # sort by position descending (farther first), we operate in-place copy
            order = np.argsort(-self.x[d][:n])
            x_sorted = self.x[d][:n][order]
            # distance to leader = leader_x - my_x
            gaps = np.empty(n, dtype=np.float32)
            gaps[0] = 1e9
            gaps[1:] = x_sorted[:-1] - x_sorted[1:]
            # unpermute to original indices
            inv = np.empty(n, dtype=np.int32)
            inv[order] = np.arange(n, dtype=np.int32)
            out = gaps[inv]
        return out

    def remove_departed(self, d: str):
        n = self.active[d]
        keep = self.x[d][:n] > 0.0
        k = int(keep.sum())
        if k < n:
            self.x[d][:k] = self.x[d][:n][keep]
            self.v[d][:k] = self.v[d][:n][keep]
            self.stops[d][:k] = self.stops[d][:n][keep]
            self.at_stopline[d][:k] = self.at_stopline[d][:n][keep]
            self.active[d] = k
        return n - k  # departed count
