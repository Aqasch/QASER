import numpy as np
from itertools import product, accumulate
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

__ham = np.load(f"mol_data/CH2_10q_geom_C_0.000_0.000_0.000;_H_1.080_0.000_0.000;_H_-0.225_1.056_0.000_jordan_wigner.npz")
eig_val = min(__ham['eigvals'])
print(f"EIG_VAL: ", eig_val)

# fake min energy + true min energy
post_process_val_10q_ch2 = -40 + (eig_val * -1)
post_process_val_8q_h2o = - 77.89106685 + 73.29410675728349

n_qub = 8

def dictionary_of_actions(num_qubits):
    dictionary = dict()
    i = 0
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
    for r, h in product(range(num_qubits),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary

def make_circuit_qiskit(action, qubits, circuit):
    ctrl = action[0]
    targ = (action[0] + action[1]) % qubits
    rot_qubit = action[2]
    rot_axis = action[3]
    if ctrl < qubits:
        circuit.cx([ctrl], [targ])
    if rot_qubit < qubits:
        if rot_axis == 1:
            circuit.rx(0, rot_qubit) # TODO: make a function and take angles
        elif rot_axis == 2:
            circuit.ry(0, rot_qubit)
        elif rot_axis == 3:
            circuit.rz(0, rot_qubit)
    return circuit


data = np.load(f'results/finalize/lbmt_cobyla_8qH2O_step_250_F0_energy_untweaked/summary_2.npy',allow_pickle=True)[()]
episodes = len(data['train'].keys())
err_list = []
rwd_list = []
cumulative_rwd_per_ep_list = []
for ep in range(100, episodes):
    err = data['train'][ep]['errors'][-1]+post_process_val_8q_h2o
    rwd = data['train'][ep]['reward'][-1]
    cumulative_rwd_per_ep = sum(data['train'][ep]['reward'])
    err_list.append(err)
    rwd_list.append(rwd)
    cumulative_rwd_per_ep_list.append(cumulative_rwd_per_ep)

cumulative_rewards = list(accumulate(rwd_list))
print(np.min(err_list), np.argmin(err_list))

# plt.plot(rwd_list, 'o', markersize = 4)
plt.plot(cumulative_rwd_per_ep_list, 'o', markersize = 4)
plt.plot(cumulative_rewards, label = 'cumulative reward')

succ_ep = np.argmin(err_list)
actions = data['train'][succ_ep]['actions']
circuit = QuantumCircuit(n_qub)
for a in actions:
    action = dictionary_of_actions(n_qub)[a]
    final_circuit = make_circuit_qiskit(action, n_qub, circuit)
gate_info = final_circuit.count_ops()
print(final_circuit)
print(final_circuit.depth())
print(gate_info)

plt.savefig('rwd_plot_h2o.png')