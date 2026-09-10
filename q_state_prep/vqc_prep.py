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

    def train(
        self,
        max_evaluations: int = 300,
        seed: int = 42,
        optimizer: str = "COBYLA", 
    ) -> ExperimentalResult:
        """
        Runs the classical-quantum optimization loop.

        Args:
            max_evaluations: Maximum number of objective-function evaluations.
            seed: Seed used to initialize the VQC parameters and optimizer.
            optimizer: Optimization algorithm. Supported values are
                       "COBYLA" and "SPSA".

        Returns:
            ExperimentalResult containing the optimization results
            and circuit metrics.
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

        if optimizer.upper() == "COBYLA":

            result = minimize(
                self._const_function,
                initial_weights,
                method='COBYLA',
                options={'maxiter': max_evaluations, 'disp': False}
            )
    
            best_weights = result.x
            best_cost = result.fun
            function_evaluations = result.nfev
            success = result.success
            status = result.status
            message = result.message

        elif optimizer == "SPSA":
            (
                best_weights,
                best_cost,
                success,
                status,
                message
            ) = self._train_spsa(
                initial_weights=initial_weights,
                max_evaluations=max_evaluations,
                seed=seed,
            )

            function_evaluations = len(self.cost_history)

        else:
            raise ValueError(
                f"Unsupported optimizer: {optimizer}. "
                "Supported optimizers are: COBYLA, SPSA."
            )

        training_time = time.perf_counter() - start_time
            
        metrics = get_circuit_metrics(self.ansatz)

        return ExperimentalResult(
            weights=best_weights,
            fidelity=1.0 - best_cost,
            cost_history=self.cost_history.copy(),

            function_evaluations=function_evaluations,
            training_time=training_time,

            success=success,
            status=status,
            message=message,

            seed=seed,
            reps=self.ansatz.metadata["reps"] if "reps" in self.ansatz.metadata else 0,

            num_qubits=metrics["num_qubits"],
            num_parameters=metrics["num_parameters"],
            num_gates=metrics["num_gates"],
            num_cnots=metrics["num_cnots"],
            depth=metrics["depth"],
            
        )

    def _train_spsa(
            self, 
            initial_weights: np.ndarray,
            max_evaluations: int,
            seed: int
    ) -> tuple[np.ndarray, float, bool, int, str]:

        rng = np.random.default_rng(seed)

        weights = initial_weights.copy()

        num_params = len(weights)

        # SPSA hyperparameters
        a = 0.1
        c = 0.1
        alpha = 0.602
        gamma = 0.101
        A = max(10, int(0.1 * max_evaluations))

        best_weights = weights.copy()
        best_cost = np.inf

        evaluations = 0
        iteration = 0

        while evaluations + 2 <= max_evaluations:

            iteration += 1

            # Random perturbation vector
            delta = rng.choice([-1.0, 1.0], size=num_params)

            #SPSA learning-rate schedules
            ak = a / ((iteration + A) ** alpha)
            ck = c / (iteration ** gamma)

            weight_plus = weights + ck * delta
            weight_minus = weights - ck * delta

            cost_plus = self._const_function(weight_plus)
            cost_minus = self._const_function(weight_minus)

            evaluations += 2

            # Keep track of the best evaluated point
            if cost_plus < best_cost:
                best_cost = cost_plus
                best_weights = weight_plus.copy()

            if cost_minus < best_cost:
                best_cost = cost_minus
                best_weights = weight_minus.copy()

            # SPSA gradient estimate
            gradient = (
                (cost_plus - cost_minus)
                / (2.0 * ck * delta)
            )

            weights = weights - ak * gradient

        if evaluations == 0:
            return (
                best_weights,
                best_cost,
                False, 
                1, 
                "SPSA could not perform an optimization step with the given evaluation budget.",
            )

        return (
            best_weights,
            best_cost,
            True, 
            0, 
            f"SPSA completed using {evaluations} function evaluations.",
        )

