import csv
from pathlib import Path

from q_state_prep.experiments.core import Experiment

def save_experiments_csv(experiments: list[Experiment], output_path: str | Path) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "n_qubits", 
        "reps", 
        "optimizer",
        "max_evaluations",
        "seed",
        "target_seed",
        "fidelity",
        "function_evaluations",
        "training_time",
        "success",
        "status",
        "num_parameters",
        "num_gates",
        "num_cnots",
        "depth",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for experiment in experiments:
            config = experiment.config
            result = experiment.result

            writer.writerow({
                "n_qubits": config.n_qubits, 
                "reps": config.reps, 
                "optimizer": config.optimizer,
                "max_evaluations": config.max_evaluations,
                "seed": config.seed,
                "target_seed": config.target_seed,
                "fidelity": result.fidelity,
                "function_evaluations": result.function_evaluations,
                "training_time": result.training_time,
                "success": result.success,
                "status": result.status,
                "num_parameters": result.num_parameters,
                "num_gates": result.num_gates,
                "num_cnots": result.num_cnots,
                "depth": result.depth,
            })