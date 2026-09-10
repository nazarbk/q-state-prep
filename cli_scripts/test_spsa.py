import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from q_state_prep.experiments.core import ExperimentConfig
from q_state_prep.experiments.runner import run_experiment


def main():

    config = ExperimentConfig(
        n_qubits=4,
        reps=4,
        optimizer="SPSA",
        max_evaluations=100,
        seed=0,
        target_seed=123,
    )

    experiment = run_experiment(config)

    result = experiment.result

    print(
        f"optimizer={config.optimizer} | "
        f"budget={config.max_evaluations} | "
        f"reps={config.reps} | "
        f"seed={config.seed} | "
        f"fidelity={result.fidelity:.6f} | "
        f"params={result.num_parameters} | "
        f"nfev={result.function_evaluations} | "
        f"time={result.training_time:.4f}s | "
        f"success={result.success}"
    )


if __name__ == "__main__":
    main()