import matplotlib.pyplot as plt
import numpy as np
from itertools import product, accumulate


plt.rcParams.update({'font.size': 14})

# def make_circuit_qiskit(action, qubits, circuit):
#     ctrl = action[0]
#     targ = (action[0] + action[1]) % qubits
#     rot_qubit = action[2]
#     rot_axis = action[3]
#     if ctrl < qubits:
#         circuit.cx([ctrl], [targ])
#     if rot_qubit < qubits:
#         if rot_axis == 1:
#             circuit.rx(0, rot_qubit) # TODO: make a function and take angles
#         elif rot_axis == 2:
#             circuit.ry(0, rot_qubit)
#         elif rot_axis == 3:
#             circuit.rz(0, rot_qubit)
#     return circuit



# def get_max_av_min(directory: str, post_process_val: float, n_qub: int):

#     nr_ep_per_seed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     max_depth = 0
#     min_depth = 0
#     av_depth = 0

#     max_CX = 0
#     min_CX = 0
#     av_CX = 0

#     for seed in range(1, 11):
#             data = np.load(f'results/finalize/{directory}/summary_{seed}.npy', allow_pickle=True)[()]
#             nr_ep_per_seed[seed - 1] = len(data['train'].keys())

#         for seed in range(1, 11):
#             data = np.load(f'results/finalize/{directory}/summary_{seed}.npy', allow_pickle=True)[()]
#             err_list = []

#             for ep in range(nr_ep_per_seed[seed - 1]):
#                 err = data['train'][ep]['errors'][-1] + post_process_val

#                 actions = data['train'][ep]['actions']
#                 circuit = QuantumCircuit(n_qub)
#                 for a in actions:
#                     action = dictionary_of_actions(n_qub)[a]
#                     final_circuit = make_circuit_qiskit(action, n_qub, circuit)
#                 gate_info = final_circuit.count_ops()

#                 n_cx = gate_info['cx']
#                 depth = final_circuit.depth()
#                 n_gates = sum(gate_info.values())
                
#                 max_val = max(max_val, )

#             succ_ep = np.argmin(err_list)
#             unsucc_ep = np.argmax(err_list)

def plot_performance_noisy():
    # Sample data (replace with your actual values)
    labels = ['#CNOT', 'Depth', '#GATE']
    x = np.arange(len(labels))  # label locations
    width = 0.35  # width of the bars

    # Data for 6qLiH
    # CX, depth, gates

    # expo
    expo_min_lih = [9, 23, 67]
    expo_avg_lih = [22.4, 33.9, 89.5]
    expo_max_lih = [39, 43, 100]

    # original
    orig_min_lih = [7, 9, 14]
    orig_avg_lih = [34.6, 37.8, 63.6]
    orig_max_lih = [58, 57, 91]

    # Data for 8qH2O
    # CX, depth, gates

    # expo
    expo_min_h2o = [11, 16, 53]
    expo_avg_h2o = [38, 41.7, 103.1]
    expo_max_h2o = [76, 75, 174]

    # original
    orig_min_h2o = [10, 8, 23]
    orig_avg_h2o = [44.9, 40.5, 80]
    orig_max_h2o = [104, 83, 161]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axs[0].grid(True, linestyle='--', zorder=0, alpha=0.7)
    axs[1].grid(True, linestyle='--', zorder=0, alpha=0.7)

    # LiH plot
    axs[0].bar(x - width/2, expo_max_lih, width, label='QASER (max)', color='skyblue', edgecolor='black', alpha=0.5, zorder=3)
    axs[0].bar(x - width/2, expo_avg_lih, width, label='QASER (avg)', color='skyblue', hatch='.', capsize=5, zorder=3)
    axs[0].bar(x - width/2, expo_min_lih, width, label='QASER (min)', color='skyblue', edgecolor='black', hatch='/', alpha=0.7, zorder=3)

    axs[0].bar(x + width/2, orig_max_lih, width, label='CRLQAS (max)', color='lightgreen', edgecolor='black', alpha=0.5, zorder=3)
    axs[0].bar(x + width/2, orig_avg_lih, width, label='CRLQAS (avg)', color='lightgreen', hatch='.', capsize=5, zorder=3)
    axs[0].bar(x + width/2, orig_min_lih, width, label='CRLQAS (min)', color='lightgreen', edgecolor='black', hatch='/', alpha=0.7, zorder=3)

    axs[0].set_title('6-LiH')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels)
    # axs[0].legend(loc='upper left', fontsize=8)

    # H2O plot
    axs[1].bar(x - width/2, expo_max_h2o, width, label='QASER (max)', color='skyblue', edgecolor='black', alpha=0.5, zorder=3)
    axs[1].bar(x - width/2, expo_avg_h2o, width, label='QASER (avg)', color='skyblue',  hatch='.', capsize=5, zorder=3)
    axs[1].bar(x - width/2, expo_min_h2o, width, label='QASER (min)', color='skyblue', edgecolor='black', hatch='/', alpha=0.7, zorder=3)

    axs[1].bar(x + width/2, orig_max_h2o, width, label='CRLQAS (max)', color='lightgreen', edgecolor='black', alpha=0.5, zorder=3)
    axs[1].bar(x + width/2, orig_avg_h2o, width, label='CRLQAS (avg)', color='lightgreen',  hatch='.', capsize=5, zorder=3)
    axs[1].bar(x + width/2, orig_min_h2o, width, label='CRLQAS (min)', color='lightgreen', edgecolor='black', hatch='/', alpha=0.7, zorder=3)

    axs[1].set_title('8-H₂O')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('bar_plot_noisy.png', dpi=300)


def get_real_min_energy(fake_min_energy, mol_data_file):
    import numpy as np
    __ham = np.load(f"mol_data/{mol_data_file}.npz")

    _, _, eigvals, energy_shift = __ham['hamiltonian'], __ham['weights'],__ham['eigvals'], __ham['energy_shift']
    min_eig = min(eigvals) + energy_shift

    print('MIN_EIG: ', min_eig)
    return fake_min_energy - min_eig


def plot_performance_agent():
    from itertools import product, accumulate

    # 8q H2O
    # directories = ['no_noise_lbmt_cobyla_8qH2O_step_250', 'no_noise_lbmt_cobyla_8qH2O_step_250_F0_energy_untweaked']
    # n_qub = 8
    # post_process_val = - 77.89106685 + 73.29410675728349

    # 10q H2O
    directories = ['no_noise_lbmt_cobyla_10qH2O_step_350', 'no_noise_lbmt_cobyla_10qH2O_step_350_F0_energy_untweaked']
    n_qub = 10
    post_process_val = get_real_min_energy(fake_min_energy=-79.16503540049368, mol_data_file='H2O_10q_geom_H_-0.021,_-0.002,_0.000;_O_0.835,_0.452,_0.000;_H_1.477,_-0.273,_0.000_jordan_wigner')

    # 6q LiH
    # directory = 'no_noise_lbmt_cobyla_6qLiH_step_100_F0_energy_untweaked'
    # n_qub = 6
    # post_process_val = get_real_min_energy(fake_min_energy=-12, mol_data_file='LiH_6q_geom_Li_0.000_0.000_0.000;_H_0.000_0.000_3.400_jordan_wigner')

    # 6q BeH2
    # directory = 'no_noise_lbmt_cobyla_6qBEH2_F0_energy_untweaked'
    # n_qub = 6
    # post_process_val = get_real_min_energy(fake_min_energy=-19.8615891832814, mol_data_file='BEH2_6q_geom_H_0.000_0.000_-1.330;_Be_0.000_0.000_0.000;_H_0.000_0.000_1.330_jordan_wigner')

    for seed in range(1, 11):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        axs[0].grid(True, linestyle='--', zorder=0, alpha=0.7)
        axs[1].grid(True, linestyle='--', zorder=0, alpha=0.7)

        axs[0].set_ylabel('Error threshold', fontsize=14)
        axs[1].set_ylabel('Cumulative reward', fontsize=14)

        axs[0].set_xlabel('Episodes', fontsize=14)
        axs[1].set_xlabel('Episodes', fontsize=14)

        linestyle_list = ['-', '--']
        nr_episodes = 10000

        for i, directory in enumerate(directories):
            if i == 0:
                label = 'CRLQAS'
            else:
                label = 'QASER'

            data = np.load(f'results/finalize/{directory}/summary_{seed}.npy',allow_pickle=True)[()]
            nr_episodes = min(len(data['train'].keys()), nr_episodes)
            rwd_list = []
            done_list = []
            cumulative_rwd_per_ep_list = []
            max_rwd = 0

            for ep in range(nr_episodes):
                rwd = data['train'][ep]['reward'][-1]
                done_thr = data['train'][ep]['done_threshold'] + post_process_val
                cumulative_rwd_per_ep = sum(data['train'][ep]['reward'])
                rwd_list.append(rwd)
                done_list.append(done_thr)
                max_rwd = max(max_rwd, rwd)
                cumulative_rwd_per_ep_list.append(cumulative_rwd_per_ep)

            cumulative_rewards = list(accumulate(rwd_list))

            axs[0].semilogy(done_list, linestyle_list[i], label=label)
            axs[1].semilogy(cumulative_rewards, linestyle_list[i], label=label)

            axs[0].legend(fontsize=12)
            
            # axs[0].tick_params(axis='both', which='minor', labelsize=14)
            # axs[1].tick_params(axis='both', which='minor', labelsize=14)

            plt.tight_layout()
            plt.savefig(f'agent_performance_{directories[0]}_{seed}.pdf', dpi=300)
            plt.savefig(f'agent_performance_{directories[0]}_{seed}.png', dpi=300)


def plot_performance_agent_clifford():
    seed = 1
    directory = 'clifford_circuit_test'

    fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    axs[0].grid(True, linestyle='--', zorder=0, alpha=0.7)
    axs[1].grid(True, linestyle='--', zorder=0, alpha=0.7)
    axs[2].grid(True, linestyle='--', zorder=0, alpha=0.7)

    axs[0].set_ylabel('gaussian_original', fontsize=14)
    axs[1].set_ylabel('gaussian_99995_5000_3000', fontsize=14)
    axs[2].set_ylabel('gaussian_99995_1000_5000', fontsize=14)

    axs[0].set_xlabel('Steps', fontsize=14)
    axs[1].set_xlabel('Steps', fontsize=14)
    axs[2].set_xlabel('Steps', fontsize=14)

    linestyle_list = ['-', '--']

    label = "ERROR"
    
    err_lists = []
    cumerr_lists = []

    nr_episodes =1800

    for directory in ['clifford_circuit_test_less_exp_d4_new_sigma', 
                    #   'clifford_circuit_test_less_exp_d4_new_sigma_99995_5000_3000', 
                    #   'clifford_circuit_test_less_exp_d4_new_sigma_99995_1000_5000'
                    ]:
        data = np.load(f'results/finalize/{directory}/summary_{seed}.npy',allow_pickle=True)[()]

        # nr_episodes = min(len(data['train'].keys()), nr_episodes)
        rwd_list = []
        done_list = []
        err_list = []
        cumulative_rwd_per_ep_list = []
        max_rwd = 0

        for ep in range(nr_episodes):
            rwd = data['train'][ep]['reward'][-1]
            done_thr = data['train'][ep]['done_threshold']
            err = data['train'][ep]['errors']

            if directory == 'clifford_circuit_test_less_exp':
                stabilizer =  data['train'][ep]['generators'][-1]

            cumulative_rwd_per_ep = sum(data['train'][ep]['reward'])
            cumulative_rwd_per_ep_last = data['train'][ep]['reward'][-1]

            rwd_list.append(rwd)
            done_list.append(done_thr)
            err_list.append(err)
            max_rwd = max(max_rwd, rwd)
            cumulative_rwd_per_ep_list.append(cumulative_rwd_per_ep_last)

        # cumulative_rewards = list(accumulate(rwd_list))

        err_list = [x for xs in err_list for x in xs]
        err_lists.append(err_list)
        cumerr_lists.append(rwd_list)
    
    axs[0].plot(err_lists[0], '.', label=label)
    # axs[1].plot(err_lists[1], '.', label=label)
    # axs[2].plot(err_lists[2], '.', label=label)

    axs[0].legend(fontsize=12)

    # axs[1].semilogy(cumulative_rewards, linestyle_list[i], label=label)

    # axs[0].tick_params(axis='both', which='minor', labelsize=14)
    # axs[1].tick_params(axis='both', which='minor', labelsize=14)

    plt.tight_layout()
    plt.savefig(f'agent_performance_reward_comparison.pdf', dpi=300)
    plt.savefig(f'agent_performance_reward_comparison.png', dpi=300)

if __name__ == "__main__":
    plot_performance_agent_clifford()