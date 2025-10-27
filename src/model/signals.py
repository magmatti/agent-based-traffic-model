from dataclasses import dataclass

@dataclass
class SignalPlan:
    cycle: float = 60.0  # seconds

    def is_green_ns(self, t: float) -> bool:
        # simple two-phase: NS green first half, EW green second half
        phase = t % self.cycle
        return phase < self.cycle/2

    def is_green_ew(self, t: float) -> bool:
        return not self.is_green_ns(t)
