# intersection_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import random
from typing import Deque, Dict, List, Tuple

# ----------------------------
#  Config i tryby sterowania
# ----------------------------

class ControlMode(Enum):
    FIXED_LIGHTS = auto()     # cykliczne światła: faza NS / faza EW
    RIGHT_PRIORITY = auto()   # pierwszeństwo z prawej (brak świateł)
    FIRST_COME = auto()       # pierwszeństwo wg kolejności przyjazdu (timestamp)
    SMART_LIGHTS = auto()     # „sprytne” światła (prosta adaptacja do długości kolejek)

@dataclass
class PhaseConfig:
    green_ns: int = 20        # długość zielonego NS w krokach
    green_ew: int = 20
    amber: int = 2            # światło żółte (przerwa bezpieczeństwa)

@dataclass
class SpawnConfig:
    # Prawdopodobieństwo pojawienia się nowego auta w danym kroku dla każdego wlotu (0..1)
    p_N: float = 0.3
    p_E: float = 0.3
    p_S: float = 0.3
    p_W: float = 0.3
    # Rozkład skrętów dla nowych aut (sumuje się do 1.0)
    turn_probs: Dict[str, float] = None  # {"L":0.2,"S":0.6,"R":0.2}

    def __post_init__(self):
        if self.turn_probs is None:
            self.turn_probs = {"L": 0.2, "S": 0.6, "R": 0.2}

@dataclass
class SimConfig:
    steps: int = 600
    seed: int = 0
    mode: ControlMode = ControlMode.FIXED_LIGHTS
    phase: PhaseConfig = PhaseConfig()
    spawn: SpawnConfig = SpawnConfig()
    # Pojemność skrzyżowania (ile manewrów łącznie może „przepuścić” w jednym ticku)
    # Dla prostoty = 2 (np. jednocześnie N<->S i E<->W, ale bez konfliktu).
    capacity_per_tick: int = 2

# ----------------------------
#  Reprezentacja pojazdu
# ----------------------------

@dataclass
class Vehicle:
    spawn_t: int
    # Kierunek skrętu: L / S / R
    turn: str

# ----------------------------
#  Stan skrzyżowania (kolejki)
# ----------------------------

class Intersection:
    """
    Wloty: N, E, S, W.
    Każdy wlot = jedna kolejka FIFO (bez wielopasmowości w tej wersji).
    Każde auto ma docelowy manewr (L/S/R).
    """
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.t = 0

        # Kolejki na wlotach
        self.queues: Dict[str, Deque[Vehicle]] = {
            "N": deque(), "E": deque(), "S": deque(), "W": deque()
        }

        # Metryki
        self.total_departures = 0
        self.wait_times: List[int] = []
        self.max_q_len: Dict[str, int] = {k: 0 for k in self.queues}
        self.stops_count = 0  # (na przyszłość; w tej wersji ~liczba aut które nie ruszyły gdy miały szansę)

        # Sterowanie światłami (jeśli używamy FIXED_LIGHTS/SMART_LIGHTS)
        self.current_phase = "NS"  # "NS" lub "EW"
        self.phase_timer = 0

        random.seed(cfg.seed)

    # ----------------------------
    #   POMOCNICZE
    # ----------------------------

    def _sample_turn(self) -> str:
        r = random.random()
        cum = 0.0
        for k in ("L", "S", "R"):
            cum += self.cfg.spawn.turn_probs[k]
            if r <= cum:
                return k
        return "S"

    def _spawn_step(self):
        p = self.cfg.spawn
        for dir_key, prob in zip(("N","E","S","W"), (p.p_N, p.p_E, p.p_S, p.p_W)):
            if random.random() < prob:
                self.queues[dir_key].append(Vehicle(self.t, self._sample_turn()))
                self.max_q_len[dir_key] = max(self.max_q_len[dir_key], len(self.queues[dir_key]))

    def _advance_phase_fixed(self):
        pc = self.cfg.phase
        self.phase_timer += 1
        if self.current_phase == "NS":
            # trzymamy zielone NS przez green_ns, potem żółte, potem EW
            if self.phase_timer >= pc.green_ns + pc.amber:
                self.current_phase = "EW"
                self.phase_timer = 0
        else:  # EW
            if self.phase_timer >= pc.green_ew + pc.amber:
                self.current_phase = "NS"
                self.phase_timer = 0

    def _choose_allowed_movements(self) -> List[Tuple[str, str]]:
        """
        Zwraca listę dopuszczonych ruchów na tym ticku: (wlot, turn).
        Logika zależna od trybu sterowania.
        Upraszczamy konflikty:
          - FIXED_LIGHTS: w fazie NS mogą jechać N i S (dowolne L/S/R), a w EW — E i W.
          - RIGHT_PRIORITY: wybieramy do capacity_per_tick w kolejności N, E, S, W
            z regułą: jeśli ktoś ma „prawo z prawej”, blokuje kolizyjne; (tu: wersja uproszczona).
          - FIRST_COME: globalnie najstarsze auta z czterech wlotów (do capacity_per_tick) bez konfliktów.
          - SMART_LIGHTS: jak FIXED_LIGHTS, ale zmiana fazy przy dużej różnicy kolejek.
        W wersji wstępnej rozstrzygamy konflikty bardzo konserwatywnie (priorytet fazy i brak kolizji krzyżowych).
        """
        allowed: List[Tuple[str,str]] = []
        cap = self.cfg.capacity_per_tick

        if self.cfg.mode in (ControlMode.FIXED_LIGHTS, ControlMode.SMART_LIGHTS):
            # Wersja konserwatywna: z danej osi bierzemy max po 1 aucie (jeśli jest).
            axis = ("N","S") if self.current_phase == "NS" else ("E","W")
            for d in axis:
                if len(allowed) >= cap: break
                if self.queues[d]:
                    allowed.append((d, self.queues[d][0].turn))
            # SMART_LIGHTS: szybka zmiana fazy jeśli druga oś ma znacznie dłuższą kolejkę
            if self.cfg.mode == ControlMode.SMART_LIGHTS:
                q_ns = len(self.queues["N"]) + len(self.queues["S"])
                q_ew = len(self.queues["E"]) + len(self.queues["W"])
                # jeśli różnica > 2 i min. krótka faza min. 5 ticków minęła — przełącz
                if self.phase_timer >= 5 and ((self.current_phase == "NS" and q_ew > q_ns + 2) or
                                              (self.current_phase == "EW" and q_ns > q_ew + 2)):
                    self.current_phase = "EW" if self.current_phase == "NS" else "NS"
                    self.phase_timer = 0

        elif self.cfg.mode == ControlMode.FIRST_COME:
            # Zbierz kandydatów (głowy kolejek) z timestampem
            heads: List[Tuple[int,str,str]] = []  # (spawn_t, dir, turn)
            for d in ("N","E","S","W"):
                if self.queues[d]:
                    heads.append((self.queues[d][0].spawn_t, d, self.queues[d][0].turn))
            heads.sort(key=lambda x: x[0])  # najstarsze pierwsze
            for _, d, trn in heads:
                if len(allowed) >= cap: break
                # tu można dodać matrycę konfliktów; w wersji minimalnej zakładamy 2 niekolizyjne ruchy max.
                allowed.append((d, trn))

        elif self.cfg.mode == ControlMode.RIGHT_PRIORITY:
            # Uproszczenie: priorytet cyklicznie N->E->S->W, ale jeśli kierunek po prawej też chce jechać — on ma pierwszeństwo.
            order = ["N","E","S","W"]
            taken = set()
            for d in order:
                if len(allowed) >= cap: break
                if not self.queues[d]:
                    continue
                # „prawa strona” dla kierunku d:
                right_of = {"N":"E", "E":"S", "S":"W", "W":"N"}[d]
                if self.queues[right_of] and right_of not in taken:
                    # oddaj pierwszeństwo temu po prawej (jeśli jeszcze nie pojechał)
                    allowed.append((right_of, self.queues[right_of][0].turn))
                    taken.add(right_of)
                elif d not in taken:
                    allowed.append((d, self.queues[d][0].turn))
                    taken.add(d)

        return allowed[:cap]

    def _depart(self, allowed: List[Tuple[str,str]]):
        """
        Zdejmij pojazdy z wlotów zgodnie z allowed; policz metryki.
        """
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
        # 1) Spawnowanie nowych aut
        self._spawn_step()

        # 2) Wybór dopuszczonych ruchów wg trybu
        allowed = self._choose_allowed_movements()

        # 3) Realizacja przejazdu
        self._depart(allowed)

        # 4) Aktualizacja faz świateł (jeśli dotyczy)
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

# ----------------------------
#  Przykład użycia
# ----------------------------

if __name__ == "__main__":
    cfg = SimConfig(
        steps=600,
        seed=42,
        mode=ControlMode.SMART_LIGHTS,
        phase=PhaseConfig(green_ns=18, green_ew=18, amber=2),
        spawn=SpawnConfig(p_N=0.35, p_E=0.25, p_S=0.35, p_W=0.25,
                          turn_probs={"L":0.25, "S":0.5, "R":0.25}),
        capacity_per_tick=2
    )
    sim = Intersection(cfg)
    stats = sim.run()
    print(stats)
