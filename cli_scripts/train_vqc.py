"""Train one VQC state-preparation experiment and save its learning curve."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q_state_prep.utils import count_cnots, generate_noise_map_state
from q_state_prep.vqc_prep import VQCStatePrep, create_ansatz


def run_training() -> None:
    n_qubits = 5
    reps = 5
    max_evaluations = 800
    target_amplitudes = generate_noise_map_state(n_qubits, seed=123)
    ansatz = create_ansatz(n_qubits, reps)
    trainer = VQCStatePrep(target_amplitudes, ansatz)
    result = trainer.train(max_evaluations=max_evaluations, seed=42)

    cnots = count_cnots(ansatz)
    print(f"Qubits: {n_qubits} | parameters: {result.num_parameters} | CNOTs: {cnots}")
    print(f"Best fidelity: {result.fidelity:.3%} after {result.function_evaluations} evaluations")

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(np.minimum.accumulate(result.cost_history), color="#2563eb", label="Best cost so far")
    axis.set(xlabel="Objective evaluation", ylabel="Cost (1 - fidelity)")
    axis.grid(alpha=0.3)
    axis.legend()
    output = ROOT / "media" / "learning_curve.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Learning curve saved to {output}")


if __name__ == "__main__":
    run_training()
