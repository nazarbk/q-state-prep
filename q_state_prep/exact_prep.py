import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from typing import List

def get_ry_angles(amplitudes: np.ndarray) -> List[List[float]]:

    n_states = len(amplitudes)
    n_qubits = int(np.log2(n_states))

    # 1. Code quality validations
    if 2**n_qubits != n_states:
        raise ValueError("The size of the amplitude vector must be a power of 2.")

    if not np.isclose(np.linalg.norm(amplitudes), 1.0):
        raise ValueError("The amplitude vector must be normalized (norm = 1).")

    angles_by_layer = []

    # 2. We work with probabilities (square of the amplitudes)
    probs = np.abs(amplitudes)**2

    # 3. We traverse the tree level by level (each level is a qubit)
    for layer in range(n_qubits):
        num_nodes = 2**layer

        block_size = n_states // num_nodes

        layer_angles = []

        for i in range(num_nodes):
            block = probs[i * block_size : (i + 1) * block_size]

            # We divide the probability between going towards |0> (upper half) or |1> (lower half)
            half = len(block) // 2
            prob_0 = np.sum(block[:half])
            prob_1 = np.sum(block[half:])
            prob_total = prob_0 + prob_1

            # 4. Calculating the angle (avoiding division by zero)
            if prob_total > 1e-10:
                # sin(theta/2) = sqrt(prob_1 / prob_total) -> theta = 2 * arcsin(...)
                theta = 2 * np.arcsin(np.sqrt(prob_1 / prob_total))
            else:
                theta = 0.0
            
            layer_angles.append(theta)

        angles_by_layer.append(layer_angles)

    return angles_by_layer

def build_exact_circuit(angles_by_layers: List[List[float]]) -> QuantumCircuit:

    n_qubits = len(angles_by_layers)
    qc = QuantumCircuit(n_qubits)
    
    for target_qubit, layer_angles in enumerate(angles_by_layers):
        if target_qubit == 0:
            # Layer 0: The root node has no dependencies (no controls)
            theta = layer_angles[0]
            if not np.isclose(theta, 0.0):
                qc.ry(theta, 0)
        else:
            # Layers > 0: Multiple nodes, requires controlled gates
            control_qubits = list(range(target_qubit))

            for i, theta in enumerate(layer_angles):
                # Small optimization: if the angle is 0, we save on the gates
                if np.isclose(theta, 0.0):
                    continue

                bin_str = format(i, f'0{target_qubit}b')

                # 1. Apply X gates to the controls that should be in |0>
                for j, bit in enumerate(bin_str):
                    if bit == '0':
                        qc.x(control_qubits[j])

                # 2. Apply multi-controlled RY rotation
                mc_ry = RYGate(theta).control(len(control_qubits), annotated=True)
                qc.append(mc_ry, control_qubits + [target_qubit])

                # 3. Uncompute the X gates
                for j, bit in enumerate(bin_str):
                    if bit == '0':
                        qc.x(control_qubits[j])

        qc.barrier()

    return qc