import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np

from q_state_prep.vqc_prep import *

def generate_target(n_qubits: int, seed: int = 123) -> np.ndarray:
    """
    Generates a reproducible random normalized quantum state.
    """

    rng = np.random.default_rng(seed)

    target = rng.normal(size=2 ** n_qubits)

    target = target / np.linalg.norm(target)

    return target

def main():

    n_qubits = 4
    reps_values = [1, 2, 3, 4, 5]

    target = generate_target(
        n_qubits=n_qubits,
        seed=123
    )

    for reps in reps_values:

        ansatz = create_ansatz(
            n_qubits=n_qubits,
            reps=reps
        )

        trainer = VQCStatePrep(
            target_amplitudes=target, 
            ansatz=ansatz
        )

        result = trainer.train(
            maxiter=300, 
            seed=42
        )

        print(
            f"reps={reps} | "
            f"fidelity={result.fidelity:.6f} | "
            f"params={result.num_parameters} | "
            f"cnots={result.num_cnots} | "
            f"depth={result.depth} | "
            f"gates={result.num_gates} | "
            f"nfev={result.function_evaluations} | "
            f"time={result.training_time:.4f}s"
        )

if __name__ == "__main__":
    main()