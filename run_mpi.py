from __future__ import annotations

from dataclasses import asdict

from mpi4py import MPI  # NEW

from traffic_sim.config import SimulationConfig
from traffic_sim.experiments.runner import run_single
from traffic_sim.io.logging_utils import setup_logging


def main() -> None:
    setup_logging()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg = SimulationConfig(
        backend="mpi",
        total_time=3000.0,
        dt=0.1,
        spawn_rate=0.5,
        max_vehicles=20000,
        random_seed=42,
    )

    result = run_single(cfg)

    # Print summary only on rank 0
    if rank == 0:
        print("=== MPI run finished ===")
        print(f"Backend: {result.backend}")
        print(f"Config: {asdict(cfg)}")
        print(f"Wall time: {result.wall_time_seconds:.4f} s")
        print(f"Vehicles completed: {result.vehicles_completed}")
        print(f"Avg travel time: {result.avg_travel_time:.2f} s")
        print(f"Avg stops/vehicle: {result.avg_stops_per_vehicle:.2f}")
        print(f"Throughput: {result.throughput_veh_per_min:.2f} veh/min")


if __name__ == "__main__":
    main()
