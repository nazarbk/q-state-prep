from q_state_prep.vqc_prep import create_ansatz, get_circuit_metrics, VQCStatePrep, ExperimentalResult
from q_state_prep.experiments import ExperimentConfig, run_experiment, run_reps_benchmark, summarize_by_reps
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

def test_run_experiment():
    config = ExperimentConfig(
        n_qubits=2, 
        reps=1,
        optimizer="COBYLA",
        max_evaluations=5,
        seed=42,
        target_seed=123,
    )

    experiment = run_experiment(config)

    assert experiment.config == config
    assert experiment.result.num_qubits == 2
    assert experiment.result.num_parameters > 0
    assert experiment.result.function_evaluations > 0
    assert 0.0 <= experiment.result.fidelity <= 1.0

def test_expetiment_reproducibility():
    config = ExperimentConfig(
        n_qubits=2, 
        reps=1,
        optimizer="COBYLA",
        max_evaluations=20,
        seed=42,
        target_seed=123,
    )

    experiment1 = run_experiment(config)
    experiment2 = run_experiment(config)


    assert experiment1.result.fidelity == experiment2.result.fidelity
    assert np.array_equal(
        experiment1.result.weights,
        experiment2.result.weights,
    )

def test_reps_benchmark():
    experiments = run_reps_benchmark(
        n_qubits=2,
        reps_values = [1, 2],
        seeds=[0, 1],
        target_seed=123,
        max_evaluations = 20, 
    )

    assert len(experiments) == 4

    for experiment in experiments:
        assert experiment.config.n_qubits == 2
        assert experiment.config.reps in [1, 2]
        assert experiment.config.seed in [0, 1]
        assert experiment.config.target_seed == 123
        assert 0.0 <= experiment.result.fidelity <= 1.0

def test_summaryze_by_reps():
    experiments = run_reps_benchmark(
        n_qubits=2,
        reps_values = [1, 2],
        seeds=[0, 1],
        target_seed=123,
        max_evaluations = 20, 
    )

    summary = summarize_by_reps(experiments)

    assert set(summary.keys()) == {1, 2}

    for reps in [1, 2]:
        assert "mean_fidelity" in summary[reps]
        assert "std_fidelity" in summary[reps]
        assert "min_fidelity" in summary[reps]
        assert "max_fidelity" in summary[reps]

        assert 0.0 <= summary[reps]["min_fidelity"] <= 1.0
        assert 0.0 <= summary[reps]["mean_fidelity"] <= 1.0
        assert 0.0 <= summary[reps]["max_fidelity"] <= 1.0

        assert (
            summary[reps]["min_fidelity"]
            <= summary[reps]["mean_fidelity"]
            <= summary[reps]["max_fidelity"]
        )