from q_state_prep.vqc_prep import create_ansatz, get_circuit_metrics, VQCStatePrep, ExperimentalResult
import numpy as np


def test_ansatz_parameters():
    n_qubits = 6
    reps = 3

    ansatz = create_ansatz(n_qubits, reps)

    assert ansatz.num_parameters == 48

def test_ansatz_qubits():
    n_qubits = 4
    reps = 2

    ansatz = create_ansatz(n_qubits, reps)

    assert ansatz.num_qubits == n_qubits

def test_circuit_metrics():
    ansatz = create_ansatz(4, reps=2)

    metrics = get_circuit_metrics(ansatz)

    assert metrics["num_qubits"] == 4
    assert metrics["num_parameters"] == ansatz.num_parameters
    assert metrics["depth"] == ansatz.depth()
    assert metrics["num_gates"] == ansatz.size()
    assert metrics["num_cnots"] == ansatz.count_ops().get('cx', 0)

def test_training_returns_experiment_result():
    target = np.array([
        1.0, 
        0.0,
        0.0,
        0.0
    ])

    ansatz = create_ansatz(2, reps=1)

    trainer = VQCStatePrep(
        target_amplitudes=target,
        ansatz=ansatz
    )

    result = trainer.train(
        maxiter=5, 
        seed=42
    )

    assert isinstance(result, ExperimentalResult)

    assert result.num_qubits == 2
    assert result.num_parameters == ansatz.num_parameters

    assert isinstance(result.weights, np.ndarray)
    assert len(result.weights) == result.num_parameters

    assert result.seed == 42

    assert result.function_evaluations > 0
    assert isinstance(result.success, bool)
    assert isinstance(result.status, int)
    assert isinstance(result.message, str)
    assert len(result.message) > 0

    assert 0.0 <= result.fidelity <= 1.0
    assert result.training_time >= 0.0

    assert len(result.cost_history) == result.function_evaluations