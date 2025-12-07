from dataclasses import dataclass, asdict
from typing import Literal, Optional


BackendName = Literal["sequential", "openmp", "mpi", "cuda"]


@dataclass
class SimulationConfig:
    # total time of simulation (seconds)
    total_time: float = 300.0
    # time step (seconds)
    dt: float = 0.1
    # cars/s/direction
    spawn_rate: float = 0.5
    max_vehicles: int = 2000
    random_seed: int = 42

    backend: BackendName = "sequential"

    # openMP
    num_threads: int = 1
    # mpi
    num_processes: int = 1
    # cuda
    gpu_block_size: int = 256

    output_dir: str = "results"
    # scenario desc
    label: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
