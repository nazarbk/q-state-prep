import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from q_state_prep.vqc_prep import (
    create_ansatz,
    VQCStatePrep,
)


target = np.array([
    1.0,
    0.0,
    0.0,
    0.0
])

ansatz = create_ansatz(
    n_qubits=2,
    reps=1
)

trainer = VQCStatePrep(
    target_amplitudes=target,
    ansatz=ansatz
)

result = trainer.train(
    maxiter=50,
    seed=42
)

print("\n=== RAW OPTIMIZER RESULT ===")

print(result)

print("\n=== OPTIMIZER INFORMATION ===")

print(f"Iterations:             {result.iterations}")
print(f"Function evaluations:   {result.function_evaluations}")
print(f"Cost history length:    {len(result.cost_history)}")

print("\n=== TRAINING ===")

print(f"Initial cost:            {result.cost_history[0]:.6f}")
print(f"Final cost:              {result.cost_history[-1]:.6f}")
print(f"Final fidelity:          {result.fidelity:.6f}")

print("\n=== CIRCUIT ===")

print(f"Qubits:                  {result.num_qubits}")
print(f"Parameters:              {result.num_parameters}")
print(f"Gates:                   {result.num_gates}")
print(f"CNOTs:                   {result.num_cnots}")
print(f"Depth:                   {result.depth}")

print("\n=== PERFORMANCE ===")

print(f"Training time:           {result.training_time:.6f} s")