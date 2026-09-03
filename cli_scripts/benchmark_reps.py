import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q_state_prep.vqc_prep import *
from q_state_prep.experiments import run_reps_benchmark, summarize_by_reps


def main():
    experiments = run_reps_benchmark(
        n_qubits=4,
        reps_values = [1, 2, 3, 4, 5],
        seeds=[0, 1, 2, 3, 4],
        target_seed=123,
        max_evaluations = 300, 
    )

    for experiment in experiments:
        config = experiment.config
        result = experiment.result

        print(
            f"reps={config.reps} | "
            f"seed={config.seed} | "
            f"fidelity={result.fidelity:.6f} | "
            f"params={result.num_parameters} | "
            f"cnots={result.num_cnots} | "
            f"depth={result.depth} | "
            f"gates={result.num_gates} | "
            f"nfev={result.function_evaluations} | "
            f"time={result.training_time:.4f}s"
        )

    summary = summarize_by_reps(experiments)

    print("\n === SUMMARY BY REPS === ")

    for reps, stats in sorted(summary.items()):
        print(
            f"reps={reps} | "
            f"mean={stats['mean_fidelity']:.6f} | "
            f"std={stats['std_fidelity']:.6f} | "
            f"min={stats['min_fidelity']:.6f} | "
            f"max={stats['max_fidelity']:.6f}"
        )

if __name__ == "__main__":
    main()