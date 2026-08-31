import sys
import os
# Add root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity
from q_state_prep.exact_prep import get_ry_angles, build_exact_circuit
from q_state_prep.utils import count_cnots

def run_benchmark():
    print(" Qubits | Amplitudes (States) | CNOT Gates (cx) ")

    for n_qubits in range(2, 7):
        n_states = 2**n_qubits

        amplitudes = np.random.rand(n_states)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)

        angles = get_ry_angles(amplitudes)
        circuit = build_exact_circuit(angles)

        cnots = count_cnots(circuit)

        print(f"   {n_qubits}    |          {n_states:3}         |       {cnots:4}        ")

def run_tolerance_becnhmark():
    print(" Tolerancy | CNOT Gates | Fidelity | CNOTs reduction")

    n_qubits = 6
    n_states = 2**n_qubits

    np.random.seed(42)
    amplitudes = np.random.rand(n_states)

    noise_mask = np.random.rand(n_states) > 0.20

    amplitudes[noise_mask] *= 0.05
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    target_sv = Statevector(amplitudes)
    angles = get_ry_angles(amplitudes)

    tolerances = [1e-7, 0.05, 0.10, 0.15, 0.20, 0.30]

    circ_base = build_exact_circuit(angles, tol=1e-7)
    cnots_base = count_cnots(circ_base)

    for tol in tolerances:
        circuit = build_exact_circuit(angles, tol=tol)
        cnots = count_cnots(circuit)

        aligned_circuit = circuit.reverse_bits()
        approx_sv = Statevector(aligned_circuit)
        fidelity = state_fidelity(target_sv, approx_sv)

        sales = 100 * (1 - (cnots / cnots_base)) if cnots_base > 0 else 0

        print(f" {tol:10.7f} | {cnots:4} | {fidelity:6.2%} | {sales:5.1f}% ")

if __name__ == "__main__":
    run_tolerance_becnhmark()

