# intersection_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import random
from typing import Deque, Dict, List, Tuple

# =========================
#   Tryby sterowania
# =========================

class ControlMode(Enum):
    FIXED_LIGHTS = auto()
    RIGHT_PRIORITY = auto()
    FIRST_COME = auto()
    SMART_LIGHTS = auto()

# =========================
#   Konfiguracje
# =========================

@dataclass
class PhaseConfig:
    green_ns: int = 20
    green_ew: int = 20
    amber: int = 2

@dataclass
class SpawnConfig:
    p_N: float = 0.3
    p_E: float = 0.3
    p_S: float = 0.3
    p_W: float = 0.3
    turn_probs: Dict[str, float] = field(
        default_factory=lambda: {"L": 0.2, "S": 0.6, "R": 0.2}
    )

@dataclass
class SimConfig:
    steps: int = 600
    seed: int = 0
    mode: ControlMode = ControlMode.FIXED_LIGHTS
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    capacity_per_tick: int = 2

    # --- NOWE: parametry czasowe i odstępy (tick ~ sekunda na start) ---
    time_step_s: float = 1.0
    desired_headway_s: float = 1.8   # minimalny odstęp czasowy między autami z TEGO SAMEGO wlotu
    clear_time_s: float = 1.2        # czas „oczyszczenia” wspólnej strefy konfliktu

    @property
    def min_headway_ticks(self) -> int:
        return max(1, int(round(self.desired_headway_s / self.time_step_s)))

    @property
    def clear_time_ticks(self) -> int:
        return max(1, int(round(self.clear_time_s / self.time_step_s)))

# =========================
#   Pojazd
# =========================

@dataclass
class Vehicle:
    spawn_t: int
    turn: str  # "L" / "S" / "R"

# =========================
#   Rdzeń skrzyżowania
# =========================

class Intersection:
    """
    Minimalny model skrzyżowania 4-wlotowego (N/E/S/W) oparty na kolejkach FIFO.
    W tej wersji: 1 pas na wlot, limit przepustowości per tick, tryby sterowania,
    oraz *wymuszanie odstępów*:
      - min_headway_ticks: odstęp między kolejnymi autami z TEGO SAMEGO wlotu
      - clear_time_ticks: odstęp czasowy korzystania z (konserwatywnie) wspólnej strefy konfliktu
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.t = 0

        # Kolejki: N, E, S, W
        self.queues: Dict[str, Deque[Vehicle]] = {
            "N": deque(), "E": deque(), "S": deque(), "W": deque()
        }

        # Metryki
        self.total_departures = 0
        self.wait_times: List[int] = []
        self.max_q_len: Dict[str, int] = {k: 0 for k in self.queues}
        self.stops_count = 0

        # Światła
        self.current_phase = "NS"  # "NS" lub "EW"
        self.phase_timer = 0

        # --- NOWE: pamięć do odstępów ---
        # Ostatni czas wyjazdu z każdego wlotu (dla headway)
        self.last_depart_time_per_dir: Dict[str, int] = {d: -10**9 for d in ("N","E","S","W")}
        # Ostatni czas użycia/styczności ze strefą konfliktu (konserwatywnie 1 strefa)
        self.last_conflict_clear_time: int = -10**9

        random.seed(cfg.seed)

    # ----------------------------
    #   Losowanie skrętu
    # ----------------------------
    def _sample_turn(self) -> str:
        r = random.random()
        cum = 0.0
        for k in ("L", "S", "R"):
            cum += self.cfg.spawn.turn_probs[k]
            if r <= cum:
                return k
        return "S"

    # ----------------------------
    #   Pojawianie się aut
    # ----------------------------
    def _spawn_step(self):
        p = self.cfg.spawn
        for dir_key, prob in zip(("N","E","S","W"), (p.p_N, p.p_E, p.p_S, p.p_W)):
            if random.random() < prob:
                self.queues[dir_key].append(Vehicle(self.t, self._sample_turn()))
                self.max_q_len[dir_key] = max(self.max_q_len[dir_key], len(self.queues[dir_key]))

    # ----------------------------
    #   Fazy świateł (fixed/smart)
    # ----------------------------
    def _advance_phase_fixed(self):
        pc = self.cfg.phase
        self.phase_timer += 1
        if self.current_phase == "NS":
            if self.phase_timer >= pc.green_ns + pc.amber:
                self.current_phase = "EW"
                self.phase_timer = 0
        else:
            if self.phase_timer >= pc.green_ew + pc.amber:
                self.current_phase = "NS"
                self.phase_timer = 0

    # ============================
    #   KONFLIKTY & ODSTĘPY
    # ============================

    # Na start: jedna konserwatywna „wspólna” strefa konfliktu dla wszystkich manewrów
    # (w kolejnej iteracji rozbijemy na kilka stref i/lub pełną macierz kolizji).
    def _conflict_zone_id(self, move: Tuple[str, str]) -> int:
        # move: (dir, turn) — np. ("N","S"), ("E","L") itd.
        return 0

    def _filter_by_headway_and_clear_time(self, candidates: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
        """
        Egzekwuje:
          - min_headway_ticks dla TEGO SAMEGO wlotu,
          - clear_time_ticks dla strefy konfliktu (tu: jedna wspólna strefa).
        Dodatkowo, „rezerwuje” czasy w trakcie filtrowania, aby w tym samym ticku
        kolejni kandydaci nie łamali odstępów (istotne przy capacity_per_tick > 1).
        """
        ok: List[Tuple[str,str]] = []
        for d, trn in candidates:
            # Headway per wlot
            if self.t - self.last_depart_time_per_dir[d] < self.cfg.min_headway_ticks:
                continue
            # Clear-time strefy konfliktu (konserwatywnie jedna strefa)
            cz = self._conflict_zone_id((d, trn))
            if self.t - self.last_conflict_clear_time < self.cfg.clear_time_ticks:
                continue
            # „Rezerwacja” (soft-apply) — żeby kolejny kandydat w tym samym ticku
            # uwzględniał już te przejazdy
            self.last_depart_time_per_dir[d] = self.t
            self.last_conflict_clear_time = self.t
            ok.append((d, trn))
            if len(ok) >= self.cfg.capacity_per_tick:
                break
        return ok

    # ----------------------------
    #   Wybór dozwolonych ruchów
    # ----------------------------
    def _choose_allowed_movements(self) -> List[Tuple[str, str]]:
        """
        Zwraca listę dopuszczonych ruchów (wlot, turn) w tym ticku,
        po czym przepuszcza je przez filtr odstępów (headway + clear-time).
        """
        proposed: List[Tuple[str,str]] = []
        cap = self.cfg.capacity_per_tick

        if self.cfg.mode in (ControlMode.FIXED_LIGHTS, ControlMode.SMART_LIGHTS):
            axis = ("N","S") if self.current_phase == "NS" else ("E","W")
            for d in axis:
                if len(proposed) >= cap: break
                if self.queues[d]:
                    proposed.append((d, self.queues[d][0].turn))
            if self.cfg.mode == ControlMode.SMART_LIGHTS:
                q_ns = len(self.queues["N"]) + len(self.queues["S"])
                q_ew = len(self.queues["E"]) + len(self.queues["W"])
                if self.phase_timer >= 5 and ((self.current_phase == "NS" and q_ew > q_ns + 2) or
                                              (self.current_phase == "EW" and q_ns > q_ew + 2)):
                    self.current_phase = "EW" if self.current_phase == "NS" else "NS"
                    self.phase_timer = 0

        elif self.cfg.mode == ControlMode.FIRST_COME:
            heads: List[Tuple[int,str,str]] = []
            for d in ("N","E","S","W"):
                if self.queues[d]:
                    heads.append((self.queues[d][0].spawn_t, d, self.queues[d][0].turn))
            heads.sort(key=lambda x: x[0])
            for _, d, trn in heads:
                if len(proposed) >= cap: break
                proposed.append((d, trn))

        elif self.cfg.mode == ControlMode.RIGHT_PRIORITY:
            order = ["N","E","S","W"]
            taken = set()
            for d in order:
                if len(proposed) >= cap: break
                if not self.queues[d]:
                    continue
                right_of = {"N":"E", "E":"S", "S":"W", "W":"N"}[d]
                if self.queues[right_of] and right_of not in taken:
                    proposed.append((right_of, self.queues[right_of][0].turn))
                    taken.add(right_of)
                elif d not in taken:
                    proposed.append((d, self.queues[d][0].turn))
                    taken.add(d)

        # --- NOWE: filtr odstępów czasowych ---
        return self._filter_by_headway_and_clear_time(proposed)

    # ----------------------------
    #   Realizacja przejazdu
    # ----------------------------
    def _depart(self, allowed: List[Tuple[str,str]]):
        for d, _turn in allowed:
            if self.queues[d]:
                v = self.queues[d].popleft()
                wait = self.t - v.spawn_t
                self.wait_times.append(wait)
                self.total_departures += 1

    # ----------------------------
    #   Jeden krok symulacji
    # ----------------------------
    def step(self):
        # 1) Nowe auta
        self._spawn_step()
        # 2) Decyzja, kto może jechać (z uwzględnieniem odstępów)
        allowed = self._choose_allowed_movements()
        # 3) Przejazd
        self._depart(allowed)
        # 4) Faza świateł
        if self.cfg.mode in (ControlMode.FIXED_LIGHTS, ControlMode.SMART_LIGHTS):
            self._advance_phase_fixed()
        # 5) Zegar +
        self.t += 1

    # ----------------------------
    #   Uruchomienie symulacji
    # ----------------------------
    def run(self) -> Dict[str, float]:
        for _ in range(self.cfg.steps):
            self.step()
        avg_wait = sum(self.wait_times)/len(self.wait_times) if self.wait_times else 0.0
        throughput = self.total_departures / max(1, self.cfg.steps)
        max_q_total = sum(self.max_q_len.values())
        return {
            "avg_wait": avg_wait,
            "throughput_per_tick": throughput,
            "total_departures": self.total_departures,
            "max_q_N": self.max_q_len["N"],
            "max_q_E": self.max_q_len["E"],
            "max_q_S": self.max_q_len["S"],
            "max_q_W": self.max_q_len["W"],
            "max_q_sum": max_q_total
        }

# =========================
#   Demo / Quick test
# =========================

if __name__ == "__main__":
    cfg = SimConfig(
        steps=600,
        seed=42,
        mode=ControlMode.SMART_LIGHTS,
        phase=PhaseConfig(green_ns=18, green_ew=18, amber=2),
        spawn=SpawnConfig(
            p_N=0.35, p_E=0.25, p_S=0.35, p_W=0.25,
            turn_probs={"L":0.25, "S":0.5, "R":0.25}
        ),
        capacity_per_tick=2,
        # Odstępy (możesz regulować):
        time_step_s=1.0,
        desired_headway_s=1.8,
        clear_time_s=1.2,
    )
    sim = Intersection(cfg)
    stats = sim.run()
    print(stats)
