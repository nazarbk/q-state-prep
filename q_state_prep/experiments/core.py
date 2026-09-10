from dataclasses import dataclass
from typing import Literal

import numpy as np

from q_state_prep.vqc_prep import ExperimentalResult

@dataclass
class ExperimentConfig:
    n_qubits: int
    reps: int
    optimizer: str
    max_evaluations: int
    seed: int
    target_seed: int

@dataclass
class Experiment:
    config: ExperimentConfig
    result: ExperimentalResult


@dataclass(frozen=True)
class ComparisonConfig:
    """Inputs shared by every method shown in an interactive comparison."""

    n_qubits: int
    reps: int
    optimizer: Literal["COBYLA", "SPSA"]
    max_evaluations: int
    optimization_seed: int
    target_seed: int
    pruning_tolerance: float


@dataclass
class MethodComparison:
    name: str
    fidelity: float
    num_cnots: int
    depth: int
    amplitudes: np.ndarray
    training_time: float
    function_evaluations: int | None = None
    cost_history: list[float] | None = None


@dataclass
class ComparisonResult:
    config: ComparisonConfig
    target_amplitudes: np.ndarray
    exact: MethodComparison
    pruned: MethodComparison
    variational: MethodComparison
