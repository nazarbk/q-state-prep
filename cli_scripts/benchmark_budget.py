import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q_state_prep.experiments import run_budget_benchmark, save_experiments_csv
from q_state_prep.experiments.plotting import plot_fidelity_vs_budget, plot_fidelity_vs_budget_std

def main():
    experiments = run_budget_benchmark(
        n_qubits=4, 
        reps_values=[1, 2, 3, 4, 5],
        seeds=[0, 1, 2, 3, 4],
        target_seed=123, 
        evaluation_budgets=[100, 300, 600, 1000],
    )

    save_experiments_csv(
        experiments,
        "media/experiments/budget_benchmark.csv",
    )

    plot_fidelity_vs_budget(experiments)
    plot_fidelity_vs_budget_std(experiments)

    for experiment in experiments:
        config = experiment.config
        result = experiment.result

        print(
            f"budget={config.max_evaluations} | "
            f"reps={config.reps} | "
            f"seed={config.seed} | "
            f"fidelity={result.fidelity:.6f} | "
            f"params={result.num_parameters} | "
            f"nfev={result.function_evaluations} | "
            f"time={result.training_time:.4f}s"
        )

if __name__ == "__main__":
    main()