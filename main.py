from traffic_sim.config import SimulationConfig
from traffic_sim.experiments.runner import run_single
from traffic_sim.io.logging_utils import setup_logging, logger
from traffic_sim.io.results_writer import save_result_as_json
from traffic_sim.backends import BACKENDS


def choose_backend() -> str:
    print("=== Choose backend ===")
    for i, name in enumerate(BACKENDS.keys(), start=1):
        print(f"{i}. {name}")
    choice = input("Enter number: ").strip()

    try:
        idx = int(choice) - 1
        name = list(BACKENDS.keys())[idx]
    except (ValueError, IndexError):
        print("Invalid choice, falling back to 'sequential'")
        name = "sequential"
    return name


def main():
    setup_logging()

    print("=== Agent-based Traffic Simulation ===")

    backend_name = choose_backend()

    try:
        total_time = float(input("Total simulation time [s] (default 300): ") or "300")
        dt = float(input("Time step dt [s] (default 0.1): ") or "0.1")
        spawn_rate = float(input("Spawn rate [veh/s/direction] (default 0.5): ") or "0.5")
        max_veh = int(input("Max vehicles in world (default 2000): ") or "2000")
    except ValueError:
        print("Invalid input, using defaults.")
        total_time, dt, spawn_rate, max_veh = 300.0, 0.1, 0.5, 2000

    cfg = SimulationConfig(
        backend=backend_name,
        total_time=total_time,
        dt=dt,
        spawn_rate=spawn_rate,
        max_vehicles=max_veh,
    )

    logger.info(f"Running simulation with backend='{backend_name}'")
    result = run_single(cfg)

    logger.info("Simulation finished.")
    logger.info(f"Wall time: {result.wall_time_seconds:.4f} s")
    logger.info(f"Vehicles completed: {result.vehicles_completed}")
    logger.info(f"Avg travel time: {result.avg_travel_time:.2f} s")
    logger.info(f"Avg stops/vehicle: {result.avg_stops_per_vehicle:.2f}")
    logger.info(f"Throughput: {result.throughput_veh_per_min:.2f} veh/min")

    path = save_result_as_json(result, cfg.output_dir)
    logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    main()
