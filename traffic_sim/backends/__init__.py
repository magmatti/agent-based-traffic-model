from typing import Dict, Type

from traffic_sim.backends.base_backend import SimulationBackend
from traffic_sim.backends.backend_sequential import SequentialBackend
from traffic_sim.backends.backend_openmp import OpenMPBackend
from traffic_sim.backends.backend_cuda import CUDABackend

# IMPORTANT
# MPIBackend is intentionally NOT loaded here
# run_mpi.py will import it directly when needed

BACKENDS: Dict[str, Type[SimulationBackend]] = {
    SequentialBackend.name: SequentialBackend,
    OpenMPBackend.name: OpenMPBackend,
    CUDABackend.name: CUDABackend
}


def get_backend(name: str) -> Type[SimulationBackend]:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {', '.join(BACKENDS.keys())}"
        )
