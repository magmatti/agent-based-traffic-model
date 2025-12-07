from abc import ABC, abstractmethod

from traffic_sim.config import SimulationConfig
from traffic_sim.metrics.types import SimulationResult


class SimulationBackend(ABC):
    """
    Abstract base for all backends (Sequential, OpenMP, MPI, CUDA).
    """

    name: str = "base"

    def __init__(self, config: SimulationConfig):
        self.config = config


    @abstractmethod
    def run(self) -> SimulationResult:
        """
        Runs simulation and returns results.
        
        :param self: Description
        :return: Description
        :rtype: SimulationResult
        """
        raise NotImplementedError
