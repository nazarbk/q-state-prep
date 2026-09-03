from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import  List
import numpy as np
import time

def create_ansatz(n_qubits: int, reps: int = 2) -> QuantumCircuit:
    """
    Creates a hardware-optimized parameterized circuit (Ansatz) using EfficientU2.

    Args:
        n_qubits: Number of qubits.
        reps: Number of times the entanglement pattern is repeated.
              The higher the reps, the higher the fidelity you can achieve, but it consumes more CNOTs.

    Returns:
        A QuantumCircuit with free parameters.
    """

    ansatz = EfficientSU2(
        num_qubits=n_qubits,
        su2_gates=['ry', 'rz'],
        entanglement='linear',
        reps=reps
    )

    return ansatz.decompose()

def get_circuit_metrics(ansatz: QuantumCircuit) -> dict:
    """
    Calculate structural metrics of a quantum circuit.

    Args:
        ansatz: Quantum circuit to analyze.

    Returns:
        Dictionary containing circuit metrics.
    """

    return {
        "num_qubits": ansatz.num_qubits,
        "num_parameters": ansatz.num_parameters,
        "depth": ansatz.depth(),
        "num_gates": ansatz.size(),
        "num_cnots": ansatz.count_ops().get('cx', 0),
    }

@dataclass
class ExperimentalResult:
    """
    Stores the results and metrics of a VQC training experiment
    """

    weights: np.ndarray
    fidelity: float
    cost_history: List[float]

    function_evaluations: int
    training_time: float
    success: bool
    status: int
    message: str

    seed: int
    reps: int

    num_qubits: int
    num_parameters: int
    num_gates: int
    num_cnots: int
    depth: int

class VQCStatePrep:
    def __init__(self, target_amplitudes: np.ndarray, ansatz: QuantumCircuit):
        """
        Initializes the variational trainer.

        Args:
            target_amplitudes: Array containing the amplitudes of the target state.
            ansatz: The parameterized Qiskit circuit.
        """

        self.target_sv = Statevector(target_amplitudes)
        self.ansatz = ansatz

        self.cost_history = []

    def _const_function(self, weights: np.ndarray) -> float:
        """
        Calculates the current error of the circuit given a set of weights.
        """

        bound_circuit = self.ansatz.assign_parameters(weights)

        current_sv = Statevector(bound_circuit)

        fid = state_fidelity(self.target_sv, current_sv)

        cost = 1.0 - fid

        self.cost_history.append(cost)

        return cost

    def train(self, maxiter: int = 300, seed: int = 42) -> ExperimentalResult:
        """
        Runs the classical-quantum optimization loop.

        Args:
            - maxiter: Maximum number of objective function evaluations.
            - seed: Seed used to initialize the VQC parameters.

        Returns:
            - best_weights: The final optimized angles.
            - best_fidelity: The maximum fidelity achieved.
            - cost_history: The list containing the history of the cost function.
        """

        num_params = self.ansatz.num_parameters

        # We initialize the angles to random values between -pi and pi
        rgn = np.random.default_rng(seed)

        initial_weights = rgn.uniform(
            -np.pi, 
            np.pi, 
            num_params
        )

        self.cost_history = []

        start_time = time.perf_counter()

        result = minimize(
            self._const_function,
            initial_weights,
            method='COBYLA',
            options={'maxiter': maxiter, 'disp': False}
        )

        training_time = time.perf_counter() - start_time

        metrics = get_circuit_metrics(self.ansatz)

        return ExperimentalResult(
            weights=result.x,
            fidelity=1.0 - result.fun,
            cost_history=self.cost_history.copy(),

            function_evaluations=result.nfev,
            training_time=training_time,

            success=result.success,
            status=result.status,
            message=result.message,

            seed=seed,
            reps=self.ansatz.metadata["reps"] if "reps" in self.ansatz.metadata else 0,

            num_qubits=metrics["num_qubits"],
            num_parameters=metrics["num_parameters"],
            num_gates=metrics["num_gates"],
            num_cnots=metrics["num_cnots"],
            depth=metrics["depth"],
            
        )
        