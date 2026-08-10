import numpy as np
from qiskit.quantum_info import Statevector
from q_state_prep.exact_prep import get_ry_angles, build_exact_circuit

def test_exact_circuit_probabilities():
    target_probs = np.array([0.1, 0.1, 0.2, 0.0, 0.1, 0.2, 0.1, 0.2])
    target_amplitudes = np.sqrt(target_probs)

    angles = get_ry_angles(target_amplitudes)
    circuit = build_exact_circuit(angles)
    aligned_circuit = circuit.reverse_bits()

    print(circuit.draw('mpl'))

    state_vector = Statevector(aligned_circuit)
    simulated_probs = state_vector.probabilities()

    np.testing.assert_allclose(simulated_probs, target_probs, atol=1e-7)
