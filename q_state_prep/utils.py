
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2

def count_cnots(qc: QuantumCircuit) -> int:
    backend = GenericBackendV2(
        num_qubits=max(2, qc.num_qubits),
        basis_gates=['cx', 'id', 'rz', 'sx', 'x']
    )

    transpiled_qc = transpile(qc, backend=backend, optimization_level=3)

    ops = transpiled_qc.count_ops()
    return ops.get('cx', 0)

def generate_noise_map_state(n_qubits: int) -> np.ndarray:
    """
    Generates a target state that simulates a 1D visual noise map,
    ideal for procedural video game environments.
    """

    n_states = 2**n_qubits
    x = np.linspace(0, 4 * np.pi, n_states)

    amplitudes = np.sin(x) + 0.5 * np.cos(2.5 * x) + 0.2 * np.random.rand(n_states)

    amplitudes = np.abs(amplitudes)

    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    # print(f"AMPLITUDES: + {amplitudes}")

    return amplitudes