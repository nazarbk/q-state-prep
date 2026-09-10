"""One reproducible comparison of exact, pruned, and variational preparation."""

from time import perf_counter

import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity

from q_state_prep.exact_prep import build_exact_circuit, get_ry_angles
from q_state_prep.experiments.core import (
    ComparisonConfig,
    ComparisonResult,
    MethodComparison,
)
from q_state_prep.utils import count_cnots, generate_noise_map_state
from q_state_prep.vqc_prep import VQCStatePrep, create_ansatz, get_circuit_metrics


def _evaluate_exact(
    name: str,
    target: np.ndarray,
    angles: list[list[float]],
    tolerance: float,
) -> MethodComparison:
    started_at = perf_counter()
    circuit = build_exact_circuit(angles, tol=tolerance)
    # The construction follows most-significant-bit first, while Qiskit stores
    # statevector indices in little-endian order.
    state = Statevector(circuit.reverse_bits())
    return MethodComparison(
        name=name,
        fidelity=float(state_fidelity(Statevector(target), state)),
        num_cnots=count_cnots(circuit),
        depth=circuit.depth(),
        amplitudes=np.abs(state.data),
        training_time=perf_counter() - started_at,
    )


def run_comparison(config: ComparisonConfig) -> ComparisonResult:
    """Run the three preparation methods against the same seeded target."""
    target = generate_noise_map_state(config.n_qubits, seed=config.target_seed)
    angles = get_ry_angles(target)

    exact = _evaluate_exact("Exact", target, angles, tolerance=0.0)
    pruned = _evaluate_exact(
        "Pruned exact", target, angles, tolerance=config.pruning_tolerance
    )

    ansatz = create_ansatz(config.n_qubits, config.reps)
    trainer = VQCStatePrep(target, ansatz)
    vqc_result = trainer.train(
        max_evaluations=config.max_evaluations,
        seed=config.optimization_seed,
        optimizer=config.optimizer,
    )
    bound_ansatz = ansatz.assign_parameters(vqc_result.weights)
    variational_state = Statevector(bound_ansatz)
    metrics = get_circuit_metrics(ansatz)
    variational = MethodComparison(
        name=f"VQC ({config.optimizer})",
        fidelity=vqc_result.fidelity,
        num_cnots=count_cnots(ansatz),
        depth=metrics["depth"],
        amplitudes=np.abs(variational_state.data),
        training_time=vqc_result.training_time,
        function_evaluations=vqc_result.function_evaluations,
        cost_history=vqc_result.cost_history,
    )

    return ComparisonResult(
        config=config,
        target_amplitudes=target,
        exact=exact,
        pruned=pruned,
        variational=variational,
    )
