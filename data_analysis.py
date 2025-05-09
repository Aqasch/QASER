import numpy as np
from itertools import product, accumulate
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


def get_real_min_energy(fake_min_energy, mol_data_file):
    __ham = np.load(f"mol_data/{mol_data_file}.npz")

    _, _, eigvals, energy_shift = __ham['hamiltonian'], __ham['weights'],__ham['eigvals'], __ham['energy_shift']
    min_eig = min(eigvals) + energy_shift

    print('MIN_EIG: ', min_eig)
    return fake_min_energy - min_eig

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
    n_cx = 0
    n_rot = 0
    ctrl = action[0]
    targ = (action[0] + action[1]) % qubits
    rot_qubit = action[2]
    rot_axis = action[3]
    if ctrl < qubits:
        n_cx += 1
        circuit.cx([ctrl], [targ])
    if rot_qubit < qubits:
        n_rot += 1
        if rot_axis == 1:
            circuit.rx(0, rot_qubit) # TODO: make a function and take angles
        elif rot_axis == 2:
            circuit.ry(0, rot_qubit)
        elif rot_axis == 3:
            circuit.rz(0, rot_qubit)
    return circuit, n_cx, n_rot


if __name__ == "__main__":
    # 8q H2O
    # directory = 'noisy_lbmt_cobyla_8qH2O_step_250_F0_energy_untweaked'
    # n_qub = 8
    # post_process_val = - 77.89106685 + 73.29410675728349

    # 10q H2O
    # directory = 'no_noise_lbmt_cobyla_10qH2O_step_350_F0_energy_untweaked'
    # n_qub = 10
    # post_process_val = get_real_min_energy(fake_min_energy=-79.16503540049368, mol_data_file='H2O_10q_geom_H_-0.021,_-0.002,_0.000;_O_0.835,_0.452,_0.000;_H_1.477,_-0.273,_0.000_jordan_wigner')

    # 6q LiH
    # directory = 'no_noise_lbmt_cobyla_6qLiH_step_100_F0_energy_untweaked'
    # n_qub = 6
    # post_process_val = get_real_min_energy(fake_min_energy=-12, mol_data_file='LiH_6q_geom_Li_0.000_0.000_0.000;_H_0.000_0.000_3.400_jordan_wigner')

    # 6q BeH2
    # directory = 'no_noise_lbmt_cobyla_6qBEH2_F0_energy_untweaked'
    # n_qub = 6
    # post_process_val = get_real_min_energy(fake_min_energy=-19.8615891832814, mol_data_file='BEH2_6q_geom_H_0.000_0.000_-1.330;_Be_0.000_0.000_0.000;_H_0.000_0.000_1.330_jordan_wigner')


    # nr_episodes = 10000
    nr_ep_per_seed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for seed in range(1, 11):
        data = np.load(f'results/finalize/{directory}/summary_{seed}.npy',allow_pickle=True)[()]
        # nr_episodes = min(len(data['train'].keys()), nr_episodes)
        nr_ep_per_seed[seed - 1] = len(data['train'].keys())

    for seed in range(1, 11):
        data = np.load(f'results/finalize/{directory}/summary_{seed}.npy',allow_pickle=True)[()]
        err_list = []
        rwd_list = []
        done_list = []
        nfev_list = []
        cumulative_rwd_per_ep_list = []
        max_rwd = 0

        for ep in range(nr_ep_per_seed[seed - 1]):
            err = data['train'][ep]['errors'][-1]
            rwd = data['train'][ep]['reward'][-1]
            time = data['train'][ep]['time'][-1]
            nfev = data['train'][ep]['nfev'][-1]
            done_thr = data['train'][ep]['done_threshold']
            cumulative_rwd_per_ep = sum(data['train'][ep]['reward'])
            nfev_list.append(nfev)
            err_list.append(err)
            rwd_list.append(rwd)
            done_list.append(done_thr)
            max_rwd = max(max_rwd, rwd)
            cumulative_rwd_per_ep_list.append(cumulative_rwd_per_ep)

        cumulative_rewards = list(accumulate(rwd_list))
        cumulative_nfevs = list(accumulate(nfev_list))

        # plt.plot(rwd_list, 'o', markersize = 4) 
        plt.semilogy(done_list, markersize = 4)
        # plt.plot(cumulative_rewards, label = 'cumulative reward')

        succ_ep = np.argmin(err_list)
        sum_time = 0
        for ep in range(succ_ep):
            time = data['train'][ep]['time'][-1] / 3600
            sum_time += time

        actions = data['train'][succ_ep]['actions']
        circuit = QuantumCircuit(n_qub)
        for a in actions:
            action = dictionary_of_actions(n_qub)[a]
            final_ circuit, n_cx, n_rot = make_circuit_qiskit(action, n_qub, circuit)
        gate_info = final_circuit.count_ops()

        print(seed, directory, np.min(err_list), np.argmin(err_list), gate_info['cx'], final_circuit.depth(), sum(cumulative_nfevs), sum(gate_info.values()), sum_time)
        # plt.legend()
        # plt.savefig('rwd_plot_done_h20_semilog.png', dpi=300)