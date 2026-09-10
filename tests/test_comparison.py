from q_state_prep.experiments import ComparisonConfig, run_comparison


def test_comparison_uses_the_same_target_for_every_method():
    comparison = run_comparison(
        ComparisonConfig(
            n_qubits=2,
            reps=1,
            optimizer="SPSA",
            max_evaluations=10,
            optimization_seed=42,
            target_seed=123,
            pruning_tolerance=0.3,
        )
    )

    assert comparison.exact.fidelity > 0.999999
    assert comparison.pruned.num_cnots <= comparison.exact.num_cnots
    assert comparison.variational.function_evaluations == 9
    assert comparison.variational.amplitudes.shape == comparison.target_amplitudes.shape
