"""Interactive Streamlit explorer for quantum state-preparation trade-offs."""

import sys
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


def pareto_frontier(methods: list) -> set[str]:
    """Return methods with no alternative that is cheaper and at least as accurate."""
    return {
        method.name
        for method in methods
        if not any(
            other is not method
            and other.num_cnots <= method.num_cnots
            and other.fidelity >= method.fidelity
            and (other.num_cnots < method.num_cnots or other.fidelity > method.fidelity)
            for other in methods
        )
    }


def render_reading_guide() -> None:
    with st.expander("How to read the experiment", expanded=True):
        st.markdown(
            "- **Target seed** fixes the state to prepare. Keep it unchanged when comparing methods.\n"
            "- **Optimizer seed** fixes the VQC starting point; changing it measures optimizer variability, not a new target.\n"
            "- **VQC layers (reps)** add trainable rotations and entangling gates. They can improve expressiveness, but also increase CNOTs and COBYLA's minimum budget.\n"
            "- **Pruning tolerance** removes small controlled rotations. Higher values usually reduce circuit cost at the expense of fidelity.\n"
            "- **Evaluation budget** only affects VQC training. If the convergence curve is still falling at the end, a larger budget may help."
        )


with st.sidebar:
    st.header("Experiment configuration")
    n_qubits = st.slider("Qubits", min_value=2, max_value=6, value=4)
    reps = st.slider("VQC layers (reps)", min_value=1, max_value=5, value=3)
    optimizer = st.selectbox(
        "Optimizer",
        options=["COBYLA"],
        help="SPSA is planned for a future release and is not available in the explorer yet.",
    )
    st.selectbox(
        "Experimental optimizer",
        options=["SPSA — coming soon"],
        disabled=True,
        help="SPSA is intentionally disabled until its behavior has been benchmarked.",
    )
    parameter_count = 2 * n_qubits * (reps + 1)
    minimum_budget = parameter_count + 2
    max_evaluations = st.slider(
        "Evaluation budget", min_value=minimum_budget, max_value=800,
        value=max(300, minimum_budget), step=1,
        help=("COBYLA requires at least parameters + 2 evaluations; this ansatz has "
              f"{parameter_count} parameters. Each increment is one objective evaluation."),
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

render_reading_guide()

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
st.caption(
    "Use this plot to verify the prepared distribution: a closer overlap with the black target "
    "means greater fidelity. Here amplitudes are non-negative, so the plot is a faithful visual summary."
)

st.subheader("Resource–accuracy trade-off")
tradeoff_figure, tradeoff_axis = plt.subplots(figsize=(7, 4))
frontier = pareto_frontier(methods)
for method in methods:
    marker = "o" if method.name in frontier else "x"
    tradeoff_axis.scatter(method.num_cnots, method.fidelity, s=85, marker=marker, label=method.name)
    tradeoff_axis.annotate(method.name, (method.num_cnots, method.fidelity), xytext=(5, 5),
                           textcoords="offset points")
tradeoff_axis.set(
    xlabel="CNOTs after transpilation (log scale)",
    ylabel="Fidelity",
    ylim=(0, 1.02),
    xscale="symlog",
)
tradeoff_axis.grid(alpha=0.25)
st.pyplot(tradeoff_figure, clear_figure=True)
st.caption(
    "The preferred region is upper-left: high fidelity with fewer CNOTs. Circular points belong to "
    "the Pareto frontier; an × is dominated by another displayed method that is at least as accurate "
    "and no more expensive."
)

st.subheader("What this run suggests")
pruned_fidelity_loss = max(0.0, 100 * (comparison.exact.fidelity - comparison.pruned.fidelity))
pruned_reduction = 100 * (1 - comparison.pruned.num_cnots / baseline_cnots) if baseline_cnots else 0.0
vqc_fidelity_gap = max(0.0, 100 * (comparison.exact.fidelity - comparison.variational.fidelity))
insight_columns = st.columns(3)
with insight_columns[0]:
    st.markdown(
        f"**Exact is the reference.** It reaches {comparison.exact.fidelity:.3%} fidelity, "
        f"but requires {comparison.exact.num_cnots} transpiled CNOTs."
    )
with insight_columns[1]:
    pruning_assessment = (
        "a low-cost approximation" if pruned_fidelity_loss <= 1 else "a visibly lossy approximation"
    )
    st.markdown(
        f"**Pruning is {pruning_assessment}.** At {comparison.config.pruning_tolerance:.2f} rad, "
        f"it saves {pruned_reduction:.1f}% CNOTs and loses {pruned_fidelity_loss:.2f} percentage points of fidelity."
    )
with insight_columns[2]:
    if vqc_fidelity_gap <= 1:
        vqc_assessment = "The current VQC setting is a strong hardware-efficient candidate."
    elif comparison.variational.name in frontier:
        vqc_assessment = "The VQC is on the displayed cost–accuracy frontier."
    else:
        vqc_assessment = "Try more reps or budget only if the convergence chart shows remaining improvement."
    st.markdown(
        f"**VQC gap: {vqc_fidelity_gap:.2f} percentage points.** {vqc_assessment}"
    )

if comparison.variational.cost_history:
    st.subheader(f"{comparison.variational.name} convergence")
    costs = np.asarray(comparison.variational.cost_history)
    convergence_figure, convergence_axis = plt.subplots(figsize=(11, 3.5))
    convergence_axis.plot(np.minimum.accumulate(costs), color="#dc2626", label="Best cost so far")
    convergence_axis.set(xlabel="Objective evaluation", ylabel="Cost (1 − fidelity)")
    convergence_axis.grid(alpha=0.25)
    convergence_axis.legend()
    st.pyplot(convergence_figure, clear_figure=True)
    quarter_start = max(0, len(costs) * 3 // 4 - 1)
    late_improvement = np.minimum.accumulate(costs)[quarter_start] - np.min(costs)
    if late_improvement > 1e-3:
        st.caption(
            "The best cost is still improving in the final quarter of evaluations; a larger budget is worth testing."
        )
    else:
        st.caption(
            "The best cost is nearly flat in the final quarter; increasing the budget alone is unlikely to help much. "
            "Test a different number of reps or optimizer next."
        )

st.caption(
    "Fidelity is calculated with ideal statevector simulation. CNOT count is obtained after "
    "transpiling each circuit to the same CX/RZ/SX/X basis; it is a circuit-cost proxy, not a noisy-hardware result."
)
