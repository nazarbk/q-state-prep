from collections import defaultdict

import numpy as np

from q_state_prep.experiments.core import Experiment

def summarize_by_reps(experiments: list[Experiment]) -> dict[int, dict[str, float]]:
    grouped = defaultdict(list)

    for experiment in experiments:
        reps = experiment.config.reps
        grouped[reps].append(experiment.result.fidelity)

    summary = {}

    for reps, fidelities in grouped.items():
        values = np.array(fidelities)

        summary[reps] = {
            "mean_fidelity": float(np.mean(values)),
            "std_fidelity": float(np.std(values)),
            "min_fidelity": float(np.min(values)),
            "max_fidelity": float(np.max(values)),
        }

    return dict(summary)