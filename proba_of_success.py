import numpy as np
from itertools import product, accumulate
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


if __name__ == "__main__":
    directory = 'lbmt_cobyla_6qLiH_step_100'
    interval_length = 200

    nr_episodes = 10000

    for seed in range(0, 1):
        data = np.load(f'results/finalize/{directory}/summary_{seed}.npy',allow_pickle=True)[()]
        nr_episodes = min(len(data['train'].keys()), nr_episodes)
    
    interval_cumulative_sum = [0.0] * (nr_episodes // interval_length)
    succ_ep_per_seed = [0] * 10
    non_succ_ep_per_seed = [0] * 10

    for seed in range(1, 11):
        data = np.load(f'results/finalize/{directory}/summary_{seed}.npy', allow_pickle=True)[()]

        interval_id = 0
        for interval_start in range(0, nr_episodes - interval_length, interval_length):

            succ_count = 0
            unsucc_count = 0

            for start in (interval_start, interval_start + interval_length)
                err = data['train'][ep]['errors'][-1] + post_process_val
                err_list.append(err)

                if err < 1.59 * 0.001:
                    succ_count += 1
                else:
                    unsucc_count += 1

            interval_cumulative_sum[interval_id] += (succ_count / unsucc_count)
            interval_id += 1

        succ_ep = np.argmin(err_list)
        most_non_succ = np.argmax(err_list)

        succ_ep_per_seed[seed - 1] = succ_ep
        non_succ_ep_per_seed[seed - 1] = most_non_succ
    
    interval_cumulative_sum  = interval_cumulative_sum / 10
    
    print(succ_ep_per_seed)
    print(non_succ_ep_per_seed)
    print(f'Interval averages: {interval_cumulative_sum}')