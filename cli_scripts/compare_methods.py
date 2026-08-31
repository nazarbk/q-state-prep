import sys
import os
# Add root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time

from qiskit.quantum_info import Statevector, state_fidelity
from q_state_prep.exact_prep import get_ry_angles, build_exact_circuit
from q_state_prep.vqc_prep import create_ansatz, VQCStatePrep
from q_state_prep.utils import count_cnots, generate_noise_map_state

def run_comparision():
    print("\n ALGORITHM COMPARISION")
    print("=" * 65)

    n_qubits = 5
    target_amplitudes = generate_noise_map_state(n_qubits)
    target_sv = Statevector(target_amplitudes)

    results = {
        "Method": [],
        "CNOTs": [],
        "Fidelity": [],
        "Time_s": []
    }

    # 1. EXACT METHOD (Pure Grover-Rudolph)
    print("[*] Evaluating Exact Method (Baseline)...")
    t0 = time()
    angles = get_ry_angles(target_amplitudes)
    exact_circuit = build_exact_circuit(angles, tol=1e-7)
    exact_cnots = count_cnots(exact_circuit)

    exact_circuit_rev = exact_circuit.reverse_bits()
    exact_fid = state_fidelity(target_sv, Statevector(exact_circuit_rev))
    t1 = time()

    results["Method"].append("Exact\n(Baseline)")
    results["CNOTs"].append(exact_cnots)
    results["Fidelity"].append(exact_fid)
    results["Time_s"].append(t1-t0)

    # 2. BRANCH PRUNING METHOD
    print("[*] Evaluating Branch Pruning Method (tol=0.30)...")
    t0 = time()
    pruning_circ = build_exact_circuit(angles, tol=0.30)
    pruning_cnots = count_cnots(pruning_circ)

    pruning_circ_rev = pruning_circ.reverse_bits()
    pruning_fid = state_fidelity(target_sv, Statevector(pruning_circ_rev))
    t1 = time()

    results["Method"].append("Branch\nPruning")
    results["CNOTs"].append(pruning_cnots)
    results["Fidelity"].append(pruning_fid)
    results["Time_s"].append(t1-t0)


    # 3. VQC METHOD (Machine Learning)
    print("[*] Evaluating VQC (3 repetitions, max 500 iterations)...")
    t0 = time()
    ansatz = create_ansatz(n_qubits, reps=5)
    vqc_cnots = count_cnots(ansatz)

    trainer = VQCStatePrep(target_amplitudes, ansatz)
    _, vqc_fid, _ = trainer.train(maxiter=800)
    t1 = time()

    results["Method"].append("VQC\nCOBYLA")
    results["CNOTs"].append(vqc_cnots)
    results["Fidelity"].append(vqc_fid)
    results["Time_s"].append(t1-t0)

    # 4. Generate the Comparision Graph
    print("\n[*] Generating the Comparision Chart...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    color_cnot = '#3b82f6'
    bars = ax1.bar(results["Method"], results["CNOTs"], color=color_cnot, alpha=0.7, width=0.5)
    ax1.set_ylabel('Number of CNOT Gates', color=color_cnot, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_cnot)
    ax1.set_yscale('log')

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight='bold')

    ax2 = ax1.twinx()
    color_fid = '#ef4444'
    ax2.plot(results["Method"], results["Fidelity"], color=color_fid, marker='o', linewidth=3, markersize=10)
    ax2.set_ylabel('Fidelity', color=color_fid, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_fid)
    ax2.set_ylim(0.8, 1.05)

    for i, txt in enumerate(results["Fidelity"]):
        ax2.annotate(f'{txt*100:.1f}%',
                    (i, results["Fidelity"][i]),
                    xytext=(0, -20),
                    textcoords='offset points',
                    ha="center", color=color_fid, fontweight='bold')

    plt.title(f"Impact of VQC on Circuit Depth ({n_qubits} Qubits)\nProcedural Noise", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.2)

    file = "../media/comparision_chart.png"
    plt.savefig(file, dpi=300, bbox_inches='tight')

    print("=" * 65)
    print("RESULTS RESUME")
    print(f"Exacto:  {results['CNOTs'][0]:4} CNOTs  | Fidelity: {results['Fidelity'][0]:.2%} | Time (s): {results['Time_s'][0]:.2}s")
    print(f"Poda:    {results['CNOTs'][1]:4} CNOTs  | Fidelity: {results['Fidelity'][1]:.2%} | Time (s): {results['Time_s'][1]:.2}s")
    print(f"VQC:     {results['CNOTs'][2]:4} CNOTs  | Fidelity: {results['Fidelity'][2]:.2%} | Time (s): {results['Time_s'][2]:.2}s")
    print(f"\n📊 Graph saved as '{file}'")

if __name__ == "__main__":
    run_comparision()
