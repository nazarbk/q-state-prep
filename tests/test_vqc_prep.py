from q_state_prep.vqc_prep import create_ansatz, get_num_parameters
from q_state_prep.utils import count_cnots

ansatz = create_ansatz(6, reps=3)
weights = get_num_parameters(6, reps=3)
cnots = count_cnots(ansatz)

print(f"Parameters to train: {weights}")
print(f"Total CNOTs: {cnots}")