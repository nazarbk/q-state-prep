import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from q_state_prep.vqc_prep import create_ansatz, get_num_parameters, VQCStatePrep
from q_state_prep.utils import count_cnots

def generate_noise_map_state(n_qubits: int) -> np.ndarray:
    """
    Generates a target state that simulates a 1D visual noise map,
    ideal for procedural video game environments.
    """

    n_states = 2**n_qubits
    x = np.linspace(0, 4 * np.pi, n_states)

    amplitudes = np.sin(x) + 0.5 * np.cos(2.5 * x) + 0.2 * np.random.rand(n_states)

    amplitudes = np.abs(amplitudes)

    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    return amplitudes

def run_training():
    print("\n STARTING TRAINING: QUANTUM MACHINE LEARNING")
    print("=" * 65)

    # 1. System Configuration
    # We'll start with 4 qubits so that training takes seconds, not minutes
    n_qubits = 4
    reps = 3
    maxiter = 500

    print(f"[*] System: {n_qubits} qubits (Space of {2**n_qubits} states)")

    # 2. Generate the target state (our procedural noise map)
    print("[*] Generating the target state: 1D visual noise map...")
    target_amplitudes = generate_noise_map_state(n_qubits)

    # 3. Building the Ansatz
    ansatz = create_ansatz(n_qubits, reps)
    cnots = count_cnots(ansatz)
    weights = get_num_parameters(n_qubits, reps)
    print(f"[*] Efficient Ansatz created: {cnots} CNOTs, {weights} free parameters.")

    # 4. Initialize and Train
    trainer = VQCStatePrep(target_amplitudes, ansatz)
    print(f"[*] Running COBYLA optimizer (Maximum iterations: {maxiter})...")

    best_weights, best_fidelity, cost_history = trainer.train(maxiter=maxiter)

    print("=" * 65)
    print(f"Training completed!")
    print(f"Best fidelity achieved: {best_fidelity:.2f}")

    # 5. Save the convergence plot
    # IMPORTANT: We use `plt.savefig` instead of `plt.show()` to avoid the
    # Tkinter error we encountered earlier. This saves a flawless PNG image.
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, color='#2563eb', linewidth=2, label='Error (1 - Fidelity)')
    plt.axhline(y=0.01, color='#dc2626', linestyle='--', label='Goal (Fidelity 99%)')
    
    plt.title(f"Learning Curve VQC\n({n_qubits} Qubits, {cnots} CNOTs)", fontsize=14)
    plt.xlabel("COBYLA iterations", fontsize=12)
    plt.ylabel("Error (Cost)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    graphic_file = "./media/learning_curve.png"
    plt.savefig(graphic_file, dpi=300, bbox_inches='tight')
    print(f"📊 Training graph saved as '{graphic_file}'")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_training()