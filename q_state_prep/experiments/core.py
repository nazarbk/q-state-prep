from dataclasses import dataclass

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