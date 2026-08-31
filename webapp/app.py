import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from q_state_prep.utils import generate_noise_map_state

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
st.header("1. Target: 1D Noise Map")

# Generate amplitudes using the original function
target_amplitudes = generate_noise_map_state(n_qubits)
n_states = len(target_amplitudes)

# 1D Profile Plot
fig_target, ax_target = plt.subplots(figsize=(10, 3))
ax_target.plot(range(n_states), target_amplitudes, marker='o', color="#2563eb", linewidth=2, markersize=6)
ax_target.fill_between(range(n_states), target_amplitudes, color="#3b82f6", alpha=0.2)

ax_target.set_title(f"Ideal Noise Profile ({n_qubits} Qubits | {n_states} States)", fontweight='bold')
ax_target.set_xlabel("Quantum State Index")
ax_target.set_ylabel("Probability Amplitude")
ax_target.grid(True, alpha=0.3)

# Render plot in Streamlit
st.pyplot(fig_target)

# RENDER PLACEHOLDER (NEXT STEPS)
if run_button:
    st.divider()
    st.info("The live VQC training and Exact Method comparision will be inserted here...")
