from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.optimize import minimize
from typing import Tuple, List
import numpy as np

def create_ansatz(n_qubits: int, reps: int = 2) -> QuantumCircuit:
    """
    Creates a hardware-optimized parameterized circuit (Ansatz).

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

def get_num_parameters(n_qubits: int, reps: int) -> int:
    """
    Calculate how many weights (parameters) our AI will need to learn.
    """

    return 2 * n_qubits * (reps + 1)

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

    def train(self, maxiter: int = 300)-> Tuple[np.ndarray, float, List[float]]:
        """
        Ejecuta el bucle de optimización clásico-cuántico.

        Returns:
            - best_weights: Los ángulos finales optimizados.
            - best_fidelity: La fidelidad máxima alcanzada.
            - cost_history: La lista con el historial de la función de coste.
        """

        num_params = self.ansatz.num_parameters

        # We initialize the angles to random values between -pi and pi
        np.random.seed(42)
        initial_weights = np.random.uniform(-np.pi, np.pi, num_params)
        self.cost_history = []

        result = minimize(
            self._const_function,
            initial_weights,
            method='COBYLA',
            options={'maxiter': maxiter, 'disp': False}
        )

        best_weights = result.x 
        best_fidelity = 1.0 - result.fun 

        return best_weights, best_fidelity, self.cost_history