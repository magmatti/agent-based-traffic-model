import json
import os
from datetime import datetime

from traffic_sim.metrics.types import SimulationResult


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_result_as_json(result: SimulationResult, output_dir: str) -> str:
    _ensure_dir(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.backend}_{ts}.json"
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, indent=2, ensure_ascii=False)

    return path
