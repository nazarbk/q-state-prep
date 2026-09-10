"""Interactive Streamlit explorer for quantum state-preparation trade-offs."""

import sys
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q_state_prep.experiments import ComparisonConfig, run_comparison
from q_state_prep.utils import generate_noise_map_state


st.set_page_config(page_title="Quantum State Preparation Explorer", page_icon="⚛️", layout="wide")
st.title("Quantum State Preparation Explorer")
st.caption(
    "Compare exact preparation, branch pruning, and a hardware-efficient VQC "
    "on the same reproducible procedural-noise target."
)


def show_amplitudes(target: np.ndarray, methods: list) -> None:
    fig, axis = plt.subplots(figsize=(11, 3.5))
    indices = np.arange(target.size)
    axis.plot(indices, target, "--", color="black", linewidth=2, label="Target")
    colors = ["#2563eb", "#f59e0b", "#dc2626"]
    for method, color in zip(methods, colors):
        axis.plot(indices, method.amplitudes, marker="o", markersize=3, linewidth=1.5,
                  color=color, label=method.name)
    axis.set(xlabel="State index", ylabel="Amplitude")
    axis.grid(alpha=0.25)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    st.pyplot(fig, clear_figure=True)


@st.cache_data(show_spinner=False)
def cached_comparison(config: ComparisonConfig):
    """Avoid repeating expensive transpilation and optimization for one setup."""
    return run_comparison(config)


with st.sidebar:
    st.header("Experiment configuration")
    n_qubits = st.slider("Qubits", min_value=2, max_value=6, value=4)
    reps = st.slider("VQC layers (reps)", min_value=1, max_value=5, value=3)
    optimizer = st.selectbox("Optimizer", options=["COBYLA", "SPSA"])
    parameter_count = 2 * n_qubits * (reps + 1)
    minimum_budget = parameter_count + 2 if optimizer == "COBYLA" else 20
    default_budget = minimum_budget + 20 * math.ceil(
        max(0, 300 - minimum_budget) / 20
    )
    max_evaluations = st.slider(
        "Evaluation budget", min_value=minimum_budget, max_value=800,
        value=min(default_budget, 800), step=20,
        help=("COBYLA requires at least parameters + 2 evaluations; this ansatz has "
              f"{parameter_count} parameters."),
    )
    pruning_tolerance = st.slider(
        "Pruning angle tolerance (radians)", min_value=0.0, max_value=1.5,
        value=0.30, step=0.05,
        help="Controlled RY rotations at or below this angle are omitted.",
    )
    target_seed = st.number_input("Target seed", min_value=0, value=123, step=1)
    optimization_seed = st.number_input("Optimizer seed", min_value=0, value=42, step=1)
    run_button = st.button("Run comparison", type="primary", use_container_width=True)

config = ComparisonConfig(
    n_qubits=n_qubits,
    reps=reps,
    optimizer=optimizer,
    max_evaluations=max_evaluations,
    optimization_seed=int(optimization_seed),
    target_seed=int(target_seed),
    pruning_tolerance=pruning_tolerance,
)

if run_button:
    with st.spinner("Simulating exact, pruned, and variational circuits..."):
        st.session_state["comparison"] = cached_comparison(config)

comparison = st.session_state.get("comparison")
if comparison is None:
    target_preview = generate_noise_map_state(n_qubits, seed=int(target_seed))
    st.subheader("Target preview")
    preview_figure, preview_axis = plt.subplots(figsize=(11, 3.5))
    preview_axis.plot(target_preview, marker="o", color="#2563eb")
    preview_axis.set(xlabel="State index", ylabel="Amplitude")
    preview_axis.grid(alpha=0.25)
    st.pyplot(preview_figure, clear_figure=True)
    st.info("Set the parameters and select **Run comparison** to evaluate all methods.")
    st.stop()

if comparison.config != config:
    st.warning("The controls have changed. The results below correspond to the last run.")

st.subheader("Results")
methods = [comparison.exact, comparison.pruned, comparison.variational]
columns = st.columns(3)
baseline_cnots = comparison.exact.num_cnots
for column, method in zip(columns, methods):
    reduction = 100 * (1 - method.num_cnots / baseline_cnots) if baseline_cnots else 0.0
    with column:
        st.markdown(f"#### {method.name}")
        st.metric("Fidelity", f"{method.fidelity:.3%}")
        st.metric("CNOTs after transpilation", method.num_cnots,
                  delta="Baseline" if method is comparison.exact else f"{reduction:.1f}% vs exact",
                  delta_color="off" if method is comparison.exact else "inverse")
        st.metric("Circuit depth", method.depth)
        st.caption(f"Wall time: {method.training_time:.3f} s")
        if method.function_evaluations is not None:
            st.caption(f"Objective evaluations: {method.function_evaluations}")

st.subheader("Prepared amplitudes")
show_amplitudes(comparison.target_amplitudes, methods)

st.subheader("Resource–accuracy trade-off")
tradeoff_figure, tradeoff_axis = plt.subplots(figsize=(7, 4))
for method in methods:
    tradeoff_axis.scatter(method.num_cnots, method.fidelity, s=85, label=method.name)
    tradeoff_axis.annotate(method.name, (method.num_cnots, method.fidelity), xytext=(5, 5),
                           textcoords="offset points")
tradeoff_axis.set(xlabel="CNOTs after transpilation", ylabel="Fidelity", ylim=(0, 1.02))
tradeoff_axis.grid(alpha=0.25)
st.pyplot(tradeoff_figure, clear_figure=True)

if comparison.variational.cost_history:
    st.subheader(f"{comparison.variational.name} convergence")
    costs = np.asarray(comparison.variational.cost_history)
    convergence_figure, convergence_axis = plt.subplots(figsize=(11, 3.5))
    convergence_axis.plot(np.minimum.accumulate(costs), color="#dc2626", label="Best cost so far")
    convergence_axis.set(xlabel="Objective evaluation", ylabel="Cost (1 − fidelity)")
    convergence_axis.grid(alpha=0.25)
    convergence_axis.legend()
    st.pyplot(convergence_figure, clear_figure=True)

st.caption(
    "Fidelity is calculated with ideal statevector simulation. CNOT count is obtained after "
    "transpiling each circuit to the same CX/RZ/SX/X basis; it is a circuit-cost proxy, not a noisy-hardware result."
)
