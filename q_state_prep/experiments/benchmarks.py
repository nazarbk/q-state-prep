import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q_state_prep.experiments.core import *
from q_state_prep.experiments.runner import *


def run_reps_benchmark(n_qubits: int, reps_values: list[int], seeds: list[int], target_seed: int, max_evaluations: int) -> list[Experiment]:

    experiments = []

    for reps in reps_values:
        for seed in seeds:
            config = ExperimentConfig(
                n_qubits=n_qubits,
                reps=reps,
                optimizer="COBYLA",
                max_evaluations=max_evaluations,
                seed=seed,
                target_seed=target_seed,
            )

            experiment = run_experiment(config)
            experiments.append(experiment)

    return experiments