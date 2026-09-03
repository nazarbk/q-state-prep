import numpy as np

from q_state_prep.vqc_prep import *
from q_state_prep.experiments.core import *

def generate_random_target(n_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    dimension = 2 ** n_qubits
    target = rng.random(dimension)

    target = target / np.linalg.norm(target)

    return target

def run_experiment(config: ExperimentConfig) -> Experiment:
    target = generate_random_target(
        n_qubits=config.n_qubits, 
        seed=config.target_seed,
    )

    ansatz = create_ansatz(
        n_qubits=config.n_qubits,
        reps=config.reps,
    )

    trainer = VQCStatePrep(
        target_amplitudes=target,
        ansatz=ansatz,
    )

    result = trainer.train(
        maxiter=config.max_evaluations,
        seed=config.seed,
    )

    return Experiment(
        config=config,
        result=result,
    )