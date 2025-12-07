from typing import Iterable, List

from traffic_sim.backends import get_backend
from traffic_sim.config import SimulationConfig
from traffic_sim.metrics.types import SimulationResult


def run_single(config: SimulationConfig) -> SimulationConfig:
    BackendCls = get_backend(config.backend)
    backend = BackendCls(config)
    return backend.run()


def run_scaling_experiment(
    base_config: SimulationConfig,
    backend_name: str,
    param_name: str,
    values: Iterable[int | float]
) -> List[SimulationResult]:
    """
    Helper: changes one parameter (e.g num_threads) and runs backend
    
    :param base_config: Description
    :type base_config: SimulationConfig
    :param backend_name: Description
    :type backend_name: str
    :param param_name: Description
    :type param_name: str
    :param values: Description
    :type values: Iterable[int | float]
    :return: Description
    :rtype: List[SimulationResult]
    """

    results: List[SimulationResult] = []
    for v in values:
        cfg_dict = base_config.to_dict()
        cfg_dict["backend"] = backend_name
        cfg_dict[param_name] = v
        cfg = SimulationConfig(**cfg_dict)  # type: ignore[arg-type]
        res = run_single(cfg)
        results.append(res)
    return results
