from typing import Dict, Type

from traffic_sim.backends.base_backend import SimulationBackend
from traffic_sim.backends.backend_sequential import SequentialBackend
from traffic_sim.backends.backend_openmp import OpenMPBackend
from traffic_sim.backends.backend_mpi import MPIBackend
from traffic_sim.backends.backend_cuda import CUDABackend

BACKENDS: Dict[str, Type[SimulationBackend]] = {
    SequentialBackend.name: SequentialBackend,
    OpenMPBackend.name: OpenMPBackend,
    MPIBackend.name: MPIBackend,
    CUDABackend.name: CUDABackend
}


def get_backend(name: str) -> Type[SimulationBackend]:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {', '.join(BACKENDS.keys())}"
        )
