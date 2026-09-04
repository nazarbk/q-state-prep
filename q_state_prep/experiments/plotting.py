from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from q_state_prep.experiments.core import Experiment
from q_state_prep.experiments.analysis import summarize_by_reps

OUTPUT_DIR = Path("media/experiments")

def plot_fidelity_vs_reps(experiments: list[Experiment],) -> None:
    summary = summarize_by_reps(experiments)

    reps = sorted(summary.keys())

    mean_fidelity = [
        summary[r]["mean_fidelity"]
        for r in reps
    ]

    std_fidelity = [
        summary[r]["std_fidelity"]
        for r in reps
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.errorbar(
        reps,
        mean_fidelity,
        yerr=std_fidelity,
        marker='o',
        capsize=5,
    )

    plt.xlabel("Ansatz repetitions (reps)")
    plt.ylabel("Mean fidelity")
    plt.title("Mean fidelity vs ansatz repetitions")

    plt.xticks(reps)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fidelity_vs_reps.png", dpi=300)
    plt.close()

def plot_training_time_vs_reps(experiments: list[Experiment]) -> None:
    summary = summarize_by_reps(experiments)
    
    reps = sorted(summary.keys())

    mean_time = [
        summary[r]["mean_time"]
        for r in reps
    ]

    std_time = [
        summary[r]["std_time"]
        for r in reps
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.errorbar(
        reps,
        mean_time,
        yerr=std_time,
        marker='o',
        capsize=5,
    )

    plt.xlabel("Ansatz repetitions (reps)")
    plt.ylabel("Mean training time (s)")
    plt.title("Mean training time vs ansatz repetitions")

    plt.xticks(reps)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_time_vs_reps.png", dpi=300)
    plt.close()

def plot_depth_vs_reps(experiments: list[Experiment]) -> None:
    summary = summarize_by_reps(experiments)
    
    reps = sorted(summary.keys())

    mean_depth = [
        summary[r]["mean_depth"]
        for r in reps
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.plot(
        reps,
        mean_depth,
        marker='o',
    )

    plt.xlabel("Ansatz repetitions (reps)")
    plt.ylabel("Circuit depth")
    plt.title("Circuit depth vs ansatz repetitions")

    plt.xticks(reps)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "depth_vs_reps.png", dpi=300)
    plt.close()

def plot_fidelity_vs_depth(experiments: list[Experiment]) -> None:
    summary = summarize_by_reps(experiments)
    
    reps = sorted(summary.keys())

    mean_fidelity = [
        summary[r]["mean_fidelity"]
        for r in reps
    ]

    depths = [
        summary[r]["mean_depth"]
        for r in reps
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.errorbar(
        depths,
        mean_fidelity,
        marker='o',
    )

    plt.xlabel("Circuit depth")
    plt.ylabel("Mean fidelity")
    plt.title("Mean fidelity vs circuit depth")

    plt.xticks(reps)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fidelity_vs_depth.png", dpi=300)
    plt.close()