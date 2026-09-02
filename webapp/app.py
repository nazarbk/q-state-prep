import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from q_state_prep.utils import generate_noise_map_state, count_cnots
from q_state_prep.exact_prep import get_ry_angles, build_exact_circuit
from q_state_prep.vqc_prep import *
from qiskit.quantum_info import Statevector, state_fidelity

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Quantum Noise Generator",
    page_icon="",
    layout="wide"
)

st.title("Quantum Procedural Noise Generator")
st.markdown("Explore how Variational Quantum Circuits (VQC) optimize state preparation for procedural generation compared to exact methods.")

# SIDEBAR CONTROLS
with st.sidebar:
    st.header("⚙️ Quantum Parameters")
    
    # Safe limit: 2 to 6 qubits to prevent browser freezing during local simulation
    n_qubits = st.slider(
        "Number of qubits",
        min_value=2, max_value=6, value=4, step=1,
        help="Defines map resolution. 4 qubits = 16 values, 6 qubits = 64 values..."
    )

    st.markdown("---")
    st.subheader("VQC Configuration")
    reps = st.slider(
        "Ansatz Layers (Reps)",
        min_value=1, max_value=5, value=3, step=1,
        help="More layers = higher potential fidelity but more CNOT gates."
    )
    maxiter = st.slider(
        "Max iterations",
        min_value=100, max_value=800, value=300, step=50,
        help="Iteration limit for the COBYLA classical optimizer."
    )

    run_button = st.button("🚀 Run Comparision", type="primary", use_container_width=True)


# MAIN PANEL: TARGET STATE VISUALIZATION
st.header("1. Target: Procedural Noise Map (1D & 2D)")

# We wrap the generator in a Streamlit cache decorator
# This prevents regenerating the random noise unless n_qubits changes
# Solve, only generates 1 map per n_qubits...
if 'target_amplitudes' not in st.session_state or st.session_state.get('last_qubits') != n_qubits:
    st.session_state['target_amplitudes'] = generate_noise_map_state(n_qubits)
    st.session_state['last_qubits'] = n_qubits

# Generate amplitudes using the cached function
target_amplitudes = st.session_state['target_amplitudes']
n_states = len(target_amplitudes)

st.subheader("1D Amplitude Profile")
fig_1d, ax_1d = plt.subplots(figsize=(10, 3))
ax_1d.plot(range(n_states), target_amplitudes, marker='o', color="#2563eb", linewidth=2, markersize=6)
ax_1d.fill_between(range(n_states), target_amplitudes, color="#3b82f6", alpha=0.2)

ax_1d.set_title(f"Ideal Noise Profile ({n_qubits} Qubits | {n_states} States)", fontweight='bold')
ax_1d.set_xlabel("Quantum State Index")
ax_1d.set_ylabel("Amplitude")
ax_1d.grid(True, alpha=0.3)

# Render plot in Streamlit
st.pyplot(fig_1d)


# RENDER PLACEHOLDER (NEXT STEPS)
if run_button:
    st.divider()
    st.header("2. Exact Method (Grover-Rudolph)")

    with st.spinner("Compiling Exact Circuit... This may take a few seconds for >5 qubits."):
        # 1. Exact Method Calculations
        target_sv = Statevector(target_amplitudes)
        angles = get_ry_angles(target_amplitudes)
        exact_circuit = build_exact_circuit(angles, tol=1e-7)

        # Count CNOTs (this is the transpiler bottleneck)
        exact_cnots = count_cnots(exact_circuit)

        # Simulate the result
        exact_circuit_rev = exact_circuit.reverse_bits()
        exact_sv = Statevector(exact_circuit_rev)
        exact_fid = state_fidelity(target_sv, exact_sv)

        # Extract resulting amplitudes (using np.abs for ploting)
        exact_result_amplitudes = np.abs(exact_sv.data)

    # 2. Display metrics (Visual Cards)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Fidelity", value=f"{exact_fid * 100:.2f}%", help="100% means a perfect match with the target state.")
    with col2:
        st.metric(label="CNOT Gates (Depth)", value=exact_cnots, delta="Baseline", delta_color="off")
    with col3:
        st.metric(label="Total Qubits", value=n_qubits)

    # 3. Comparative Plot: Target VS Exact
    fig_exact, ax_exact = plt.subplots(figsize=(10, 3))

    # Target line (Ideal) 
    ax_exact.plot(range(n_states), target_amplitudes, linestyle='--', color='gray', label='Target (Ideal)', linewidth=2)

    # Exact result line
    ax_exact.plot(range(n_states), exact_result_amplitudes, marker='x', color='#ef4444', label='Exact Method', linewidth=2, alpha=0.8)

    ax_exact.set_title("Target vs Exact Method Comparision", fontweight='bold')
    ax_exact.set_xlabel("Quantum State Index")
    ax_exact.set_ylabel("Amplitude")
    ax_exact.legend()
    ax_exact.grid(True, alpha=0.3)

    st.pyplot(fig_exact)

    # 3. VARIATIONAL QUANTUM CIRCUIT (VQC)
    st.divider()
    st.header("3. Variational Quantum Circuit (VQC)")

    with st.spinner("Trainning VQC with COBYLA optimizer... This might take a few seconds."):
        # 1. Build a measure Ansatz
        ansatz = create_ansatz(n_qubits, reps)
        vqc_cnots = count_cnots(ansatz)

        # 2. Train the VQC
        trainer = VQCStatePrep(target_amplitudes, ansatz)
        result = trainer.train()

        weights = result.weights
        fidelity = result.fidelity
        cost_history = result.cost_history

        # 3. Simulate final state with the optimized weights
        bound_circuit = ansatz.assign_parameters(weights)
        vqc_sv = Statevector(bound_circuit)
        vqc_result_amplitudes = np.abs(vqc_sv.data)

        # Calculate CNOT reduction percentage
        cnot_reduction = 100 * (1 - (vqc_cnots / exact_cnots)) if exact_cnots > 0 else 0

    # 4. Display VQC Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="VQC Fidelity", value=f"{fidelity * 100:.2f}%", help="Higher is better")
    with col2:
        st.metric(label="VQC CNOT Gates", value=vqc_cnots, delta=f"-{cnot_reduction:.1f}% vs Exact", delta_color="inverse")
    with col3:
        st.metric(label="Final Error (Cost)", value=f"{cost_history[-1]:.4f}")

    # 5. Comparative Plot: Target vs VQC Output
    fig_vqc, ax_vqc = plt.subplots(figsize=(10, 3))

    # Target line (Ideal) 
    ax_vqc.plot(range(n_states), target_amplitudes, linestyle='--', color='gray', label='Target (Ideal)', linewidth=2)

    # Exact result line
    ax_vqc.plot(range(n_states), vqc_result_amplitudes, marker='x', color='#ef4444', label='VQC Method', linewidth=2, alpha=0.8)

    ax_vqc.set_title(f"Target vs VQC State Preparation ({fidelity * 100:.1f}% Fidelity)", fontweight='bold')
    ax_vqc.set_xlabel("Quantum State Index")
    ax_vqc.set_ylabel("Amplitude")
    ax_vqc.legend()
    ax_vqc.grid(True, alpha=0.3)

    st.pyplot(fig_vqc)

    # 6. Learning Curve Plot
    st.subheader("Training Convergence")
    fig_loss, ax_loss = plt.subplots(figsize=(10, 3))

    ax_loss.plot(cost_history, color="#2563eb", linewidth=2, label='Error (1 - Fidelity)')
    ax_loss.axhline(y=0.01, color='#dc2626', linestyle='--', label='Goal (99% Fidelity)')

    ax_loss.set_title("VQC Learning Curve (COBYLA)", fontweight='bold')
    ax_loss.set_xlabel("Iterations")
    ax_loss.set_ylabel("Cost")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    st.pyplot(fig_loss)