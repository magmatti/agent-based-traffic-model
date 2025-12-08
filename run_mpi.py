from __future__ import annotations

from dataclasses import asdict

from mpi4py import MPI

from traffic_sim.config import SimulationConfig
from traffic_sim.io.logging_utils import setup_logging, logger
from traffic_sim.io.results_writer import save_result_as_json
from traffic_sim.backends.backend_mpi import MPIBackend


def main() -> None:
    # Initialize logging (each rank gets the same config; we will log only on rank 0)
    setup_logging()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Hard-coded config for MPI experiments
    cfg = SimulationConfig(
        backend="mpi",
        total_time=3000.0,   # total simulated time [s]
        dt=0.1,              # time step [s]
        spawn_rate=0.5,      # vehicles per second per direction
        max_vehicles=20000,  # max vehicles in world per rank
        random_seed=42,
    )

    # Create MPI backend directly, do NOT use run_single / get_backend
    backend = MPIBackend(cfg)
    result = backend.run()

    # Only rank 0 prints and saves results
    if rank == 0:
        logger.info("=== MPI run finished ===")
        logger.info(f"Backend: {result.backend}")
        logger.info(f"Config: {asdict(cfg)}")
        logger.info(f"Wall time: {result.wall_time_seconds:.4f} s")
        logger.info(f"Vehicles completed: {result.vehicles_completed}")
        logger.info(f"Avg travel time: {result.avg_travel_time:.2f} s")
        logger.info(f"Avg stops/vehicle: {result.avg_stops_per_vehicle:.2f}")
        logger.info(f"Throughput: {result.throughput_veh_per_min:.2f} veh/min")

        path = save_result_as_json(result, cfg.output_dir)
        logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    main()
