import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q_state_prep.vqc_prep import *
from q_state_prep.experiments import run_reps_benchmark, summarize_by_reps
from q_state_prep.experiments.plotting import (plot_depth_vs_reps, plot_fidelity_vs_reps, plot_fidelity_vs_depth,  plot_training_time_vs_reps)


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
    
    plot_fidelity_vs_reps(experiments)
    plot_training_time_vs_reps(experiments)
    plot_depth_vs_reps(experiments)
    plot_fidelity_vs_depth(experiments)

    print("\n === SUMMARY BY REPS === ")

    print(
        f"{'reps':>4} | "
        f"{'fid_mean':>10} | "
        f"{'fid_std':>9} | "
        f"{'fid_min':>10} | "
        f"{'fid_max':>10} | "
        f"{'time_mean':>10} | "
        f"{'time_std':>9} | "
        f"{'nfev_mean':>10} | "
        f"{'params':>6} | "
        f"{'CNOTs':>5} | "
        f"{'depth':>5} | "
        f"{'gates':>5}"
    )

    print("-" * 130)

    for reps, stats in sorted(summary.items()):
        print(
            f"{reps:>4} | "
            f"{stats['mean_fidelity']:>10.6f} | "
            f"{stats['std_fidelity']:>9.6f} | "
            f"{stats['min_fidelity']:>10.6f} | "
            f"{stats['max_fidelity']:>10.6f} | "
            f"{stats['mean_time']:>10.4f} | "
            f"{stats['std_time']:>9.4f} | "
            f"{stats['mean_function_evaluations']:>10.1f} | "
            f"{stats['mean_parameters']:>6.1f} | "
            f"{stats['mean_cnots']:>5.1f} | "
            f"{stats['mean_depth']:>5.1f} | "
            f"{stats['mean_gates']:>5.1f}"

        )

if __name__ == "__main__":
    main()