import numpy as np
from itertools import product, accumulate
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


post_process_val_8q_h2o = - 77.89106685 + 73.29410675728349

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


if __name__ == "__main__":
    data = np.load(f'results/finalize/8qH2O_F0/summary_1.npy',allow_pickle=True)[()]
    episodes = len(data['train'].keys())
    err_list = []
    rwd_list = []
    cumulative_rwd_per_ep_list = []
    max_rwd = 0
    for ep in range(episodes):
        err = data['train'][ep]['errors'][-1]+post_process_val_8q_h2o
        rwd = data['train'][ep]['reward'][-1]
        cumulative_rwd_per_ep = sum(data['train'][ep]['reward'])
        err_list.append(err)
        rwd_list.append(rwd)
        max_rwd = max(max_rwd, rwd)
        cumulative_rwd_per_ep_list.append(cumulative_rwd_per_ep)

    cumulative_rewards = list(accumulate(rwd_list))
    print(np.min(err_list), np.argmin(err_list))

    # plt.plot(rwd_list, 'o', markersize = 4)
    print(max_rwd)

    plt.semilogy(cumulative_rwd_per_ep_list, 'o', markersize = 4)
    plt.plot(cumulative_rewards, label = 'cumulative reward')

    succ_ep = np.argmin(err_list)
    actions = data['train'][succ_ep]['actions']
    circuit = QuantumCircuit(8)
    for a in actions:
        action = dictionary_of_actions(8)[a]
        final_circuit = make_circuit_qiskit(action, 8, circuit)
    gate_info = final_circuit.count_ops()
    print(final_circuit)
    print(final_circuit.depth())
    print(gate_info)

    plt.legend()
    # plt.savefig('rwd_plot_h20.png')
    plt.savefig('rwd_plot_h20_semilog.pdf', dpi=300)