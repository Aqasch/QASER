import numpy as np
from itertools import product
from qiskit import QuantumCircuit

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


data = np.load(f'results/finalize/lbmt_cobyla_8qH2O_step_250_F0_energy_untweaked/summary_1.npy',allow_pickle=True)[()]
episodes = len(data['train'].keys())
err_list = []
for ep in range(100, episodes):
    err = data['train'][ep]['errors'][-1]+post_process_val_8q_h2o
    err_list.append(err)

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


