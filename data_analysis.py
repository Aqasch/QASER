import numpy as np

post_process_val_8q_h2o = - 77.89106685 + 73.29410675728349

data = np.load(f'results/finalize/lbmt_cobyla_8qH2O_step_250_F0_energy_untweaked/summary_1.npy',allow_pickle=True)[()]
            
episodes = len(data['train'].keys())
err_list = []
for ep in range(100, episodes):
    err = data['train'][ep]['errors'][-1]+post_process_val_8q_h2o
    err_list.append(err)

print(min(err_list), np.argmin(err_list))