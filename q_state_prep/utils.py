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