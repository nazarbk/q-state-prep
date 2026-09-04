from collections import defaultdict

import numpy as np

from q_state_prep.experiments.core import Experiment

def summarize_by_reps(experiments: list[Experiment]) -> dict[int, dict[str, float]]:
    grouped = defaultdict(list)

    for experiment in experiments:
        reps = experiment.config.reps
        grouped[reps].append(experiment)

    summary = {}

    for reps, reps_experiments in grouped.items():
        fidelities = np.array([experiment.result.fidelity for experiment in reps_experiments])

        training_times = np.array([experiment.result.training_time for experiment in reps_experiments])

        function_evaluations = np.array([experiment.result.function_evaluations for experiment in reps_experiments])

        num_parameters = np.array([experiment.result.num_parameters for experiment in reps_experiments])

        num_cnots = np.array([experiment.result.num_cnots for experiment in reps_experiments])

        depths = np.array([experiment.result.depth for experiment in reps_experiments])

        num_gates = np.array([experiment.result.num_gates for experiment in reps_experiments])

        summary[reps] = {
            "mean_fidelity": float(np.mean(fidelities)),
            "std_fidelity": float(np.std(fidelities)),
            "min_fidelity": float(np.min(fidelities)),
            "max_fidelity": float(np.max(fidelities)),
            "mean_time": float(np.mean(training_times)),
            "std_time": float(np.std(training_times)),
            "mean_function_evaluations": float(np.mean(function_evaluations)),
            "mean_parameters": float(np.mean(num_parameters)),
            "mean_cnots": float(np.mean(num_cnots)),
            "mean_depth": float(np.mean(depths)),
            "mean_gates": float(np.mean(num_gates)),
        }

    return dict(summary)