import torch
import stim
from qulacs.gate import CNOT, RX, RY, RZ
from utils import *
from sys import stdout
import scipy
import clifford_simulation as vc
import os
import numpy as np
import copy
import cirq
import curricula
from collections import Counter

try:
    from qulacs import QuantumStateGpu as QuantumState
except ImportError:
    from qulacs import QuantumState

from qulacs import ParametricQuantumCircuit

import copy
import time

def get_d5_stabilizers():
    xx = [
        [0, 1, 5, 6],
        [2, 3, 7, 8],
        [6, 7, 11, 12],
        [8, 9, 13, 14],
        [10, 11, 15, 16],
        [12, 13, 17, 18],
        [16, 17, 21, 22],
        [18, 19, 23, 24],
        [1, 2],
        [3, 4],
        [22, 23],
        [20, 21],
        [0, 5, 10, 15, 20]
    ]

    zz = [
        [1, 2, 6, 7],
        [3, 4, 8, 9],
        [5, 6, 10, 11],
        [7, 8, 12, 13],
        [11, 12, 16, 17],
        [13, 14, 18, 19],
        [15, 16, 20, 21],
        [17, 18, 22, 23],
        [0, 5],
        [10, 15],
        [9, 14],
        [19, 24]
    ]

    template = "_" * 25

    stabilizers = []
    for stabi in xx:
        stab = template
        for pos in stabi:
            stab = stab[:pos] + "X" + stab[pos + 1 :]
        stabilizers.append(stab)

    for stabi in zz:
        stab = template
        for pos in stabi:
            stab = stab[:pos] + "Z" + stab[pos + 1 :]
        stabilizers.append(stab)

    return [stim.PauliString(stab) for stab in stabilizers]


def get_d7_stabilizers():
    zz = [
        [0,1],
        [2,3],
        [4,5],
        [1,2,8,9],
        [3,4,10,11],
        [5,6,12,13],
        [7,8,14,15],
        [9,10,16,17],
        [11,12,18,19],
        [15,16,22,23],
        [17,18,24,25],
        [19,20,26,27],
        [21,22,28,29],
        [23,24,30,31],
        [25,26,32,33],
        [29,30,36,37],
        [31,32,38,39],
        [33,34,40,41],
        [35,36,42,43],
        [37,38,44,45],
        [39,40,46,47],
        [43,44],
        [45,46],
        [47,48],
    ]

    xx = [
        [0,1,7,8],
        [2,3,9,10],
        [4,5,11,12],
        [6,13],
        [7,14],
        [8,9,15,16],
        [10,11,17,18],
        [12,13,19,20],
        [14,15,21,22],
        [16,17,23,24],
        [18,19,25,26],
        [20,27],
        [21,28],
        [22,23,29,30],
        [24,25,31,32],
        [26,27,33,34],
        [28,29,35,36],
        [30,31,37,38],
        [32,33,39,40],
        [34,41],
        [35,42],
        [36,37,43,44],
        [38,39,45,46],
        [40,41,47,48],
        [0,1,2,3,4,5,6],
    ]

    template = "_" * 49

    stabilizers = []
    for stabi in xx:
        stab = template
        for pos in stabi:
            stab = stab[:pos] + "X" + stab[pos + 1 :]
        stabilizers.append(stab)

    for stabi in zz:
        stab = template
        for pos in stabi:
            stab = stab[:pos] + "Z" + stab[pos + 1 :]
        stabilizers.append(stab)

    return [stim.PauliString(stab) for stab in stabilizers]


def get_d3_stabilizers():
    return [
        stim.PauliString("ZZ_______"),
        stim.PauliString("_ZZ_ZZ___"),
        stim.PauliString("___ZZ_ZZ_"),
        stim.PauliString("_______ZZ"),
        stim.PauliString("XX_XX____"),
        stim.PauliString("__X__X___"),
        stim.PauliString("____XX_XX"),
        stim.PauliString("___X__X__"),
        stim.PauliString("XXX______"),#op logic
    ]


def get_d4_stabilizers():
    xx = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [5, 6, 9, 10],
        [10, 11, 14, 15],
        [4, 8],
        [7, 11],
        [0, 1, 2, 3]
    ]

    zz = [
        [1, 2, 5, 6],
        [4, 5, 8, 9],
        [6, 7, 10, 11],
        [9, 10, 13, 14],
        [0, 1],
        [2, 3],
        [12, 13],
        [14, 15]
    ]

    template = "_" * 16

    stabilizers = []
    for stabi in xx:
        stab = template
        for pos in stabi:
            stab = stab[:pos] + "X" + stab[pos + 1 :]
        stabilizers.append(stab)

    for stabi in zz:
        stab = template
        for pos in stabi:
            stab = stab[:pos] + "Z" + stab[pos + 1 :]
        stabilizers.append(stab)

    return [stim.PauliString(stab) for stab in stabilizers]


def get_st_stabilizers():
    return [
            stim.PauliString("___XXXX"),
            stim.PauliString("_XX__XX"),
            stim.PauliString("X_X_X_X"),
            stim.PauliString("___ZZZZ"),
            stim.PauliString("_ZZ__ZZ"),
            stim.PauliString("Z_Z_Z_Z"),
            stim.PauliString("ZZZZZZZ")
    ]

def get_random_graph_state():
    # 17 CZ gates
    return [
            stim.PauliString("X_ZZ_Z_ZZ"),
            stim.PauliString("_X__Z__Z_"),
            stim.PauliString("Z_X____ZZ"),
            stim.PauliString("Z__XZZ_ZZ"),
            stim.PauliString("_Z_ZXZ__Z"),
            stim.PauliString("Z__ZZXZZ_"),
            stim.PauliString("_____ZX__"),
            stim.PauliString("ZZZZ_Z_X_"),
            stim.PauliString("Z_ZZZ___X")

    ]

def get_five_qubit_code():
    return [
        stim.PauliString("XZZX_"),
        stim.PauliString("_XZZX"),
        stim.PauliString("X_XZZ"),
        stim.PauliString("ZX_XZ"),
        stim.PauliString("ZZZZZ")
]


class CircuitEnv():

    def __init__(self, conf, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']

        self.xyz_val1 = float(conf['env']['xyz_val1'])
        self.xyz_div1 = float(conf['env']['xyz_div1'])
        
        self.n_shots =   conf['env']['n_shots'] 
      
        self.geometry = conf['problem']['geometry'].replace(" ", "_")
        
        self.fn_type = conf['env']['fn_type']
        
        if "cnot_rwd_weight" in conf['env'].keys():
            self.cnot_rwd_weight = conf['env']['cnot_rwd_weight']
        else:
            self.cnot_rwd_weight = 1.
        
        self.current_number_of_cnots = 0

        if "boost" in conf['env'].keys():
            self.boost = int(conf['env']['boost'])
        else:
            self.boost = 0
        
        self.curriculum_dict = {}

        self.min_eig = 1.0
        self.curriculum_dict[self.geometry[-3:]] = curricula.__dict__[conf['env']['curriculum_type']](conf['env'], target_energy=self.min_eig)
     
        self.device = device
        
        self.done_threshold = conf['env']['accept_err']
        self.max_len = 10000

        self.max_cost = 0
        self.CNOT_WEIGHT = 0
        self.H_WEIGHT = 0
        self.current_len = 0
        self.current_cost = 0
        self.current_degree = 0

        original_stabilizers = get_random_graph_state()
        print(original_stabilizers)

        tbl = stim.Tableau.from_stabilizers(original_stabilizers)
        circ = tbl.to_circuit()


        s = stim.TableauSimulator()
        s.do_circuit(circ)
        self.original_stabilizers = [str(ps) for ps in s.canonical_stabilizers()]

        self.max_hamming = float(sum([len(s) for s in self.original_stabilizers]))
        self.min_hamming = 1000000.0

        print('MAX HAMMING computed in constructor: ', self.max_hamming)

        """
        Used for the reward function
        """
        max_hamming_int = int(self.max_hamming)
        param_ctr_idx = max_hamming_int // 2
        distance_arr = np.arange(0, max_hamming_int, 1) / max_hamming_int
        self.param_center = distance_arr[param_ctr_idx]
        self.param_sigma = np.std(distance_arr)

        print(self.original_stabilizers)

        if self.fn_type in ['F0_energy_depth_up']:
            self.kickstart_depth = conf['env']['kickstart_depth']

        if self.fn_type in ['F0_energy_untweaked', 'F0_hamming', 'F0_matched']:
            self.max_cost = float(conf['env']['max_cost'])
            self.CNOT_WEIGHT = float(conf['env']['cnot_weight'])
            self.H_WEIGHT = float(conf['env']['h_weight'])

        self.state_size = self.num_layers*self.num_qubits*(self.num_qubits)

        self.step_counter = -1
        self.prev_energy = 1
        self.prev_hamming = 100
        self.moments = [0]*self.num_qubits
        self.illegal_actions = [[]]*self.num_qubits

        # self.action_size = (self.num_qubits*(self.num_qubits))
        self.action_size = len(dictionary_of_cz_actions(self.num_qubits).keys())

        self.previous_action = [0, 0, 0, 0]
 
    
    def get_cirq_circuit(self):
        circuit = self.make_circuit()
        total_gate_count = circuit.get_gate_count()
        
        cirq_circuit = cirq.Circuit()
        cirq_qubits = [cirq.NamedQubit(str(q)) for q in range(self.num_qubits)]

        for i in range(total_gate_count):
            the_gate = circuit.get_gate(i)
            name_of_gate = the_gate.get_name()

            if name_of_gate != 'CNOT':
                cirq_qubit = cirq_qubits[the_gate.get_target_index_list()[0]]
                cirq_circuit.append(cirq.H.on(cirq_qubit))
            else:
                ctrl_qubit = cirq_qubits[the_gate.get_control_index_list()[0]]
                tgt_qubit = cirq_qubits[the_gate.get_target_index_list()[0]]
                cirq_circuit.append(cirq.CNOT(ctrl_qubit, tgt_qubit))
        
        return cirq_circuit
        
    def step(self, action, train_flag = True) :
        
        """
        Action is performed on the first empty layer.
        
        
        Variable 'step_counter' points last non-empty layer.
        """  
        next_state = self.state.clone()
        
        self.step_counter += 1

        """
        First two elements of the 'action' vector describes position of the CNOT gate.
        Position of rotation gate and its axis are described by action[2] and action[3].
        When action[0] == num_qubits, then there is no CNOT gate.
        When action[2] == num_qubits, then there is no Rotation gate.
        """
        
        ctrl = action[0]
        targ = (action[0] + action[1]) % self.num_qubits
        # rot_qubit = action[2]        
        
        self.action = action

        # if rot_qubit < self.num_qubits:
        #     gate_tensor = self.moments[ rot_qubit ]
        if ctrl < self.num_qubits:
            gate_tensor = max( self.moments[ctrl], self.moments[targ] )

        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
        # elif rot_qubit < self.num_qubits:
        #     next_state[gate_tensor][self.num_qubits][rot_qubit] = 1

        # if rot_qubit < self.num_qubits:
        #     self.moments[ rot_qubit ] += 1
        # elif ctrl < self.num_qubits:
        #     max_of_two_moments = max( self.moments[ctrl], self.moments[targ] )
        #     self.moments[ctrl] = max_of_two_moments +1
        #     self.moments[targ] = max_of_two_moments +1
            
        self.current_action = action
        self.illegal_action_new()

        self.state = next_state.clone()

        # self.current_circuit = self.get_cirq_circuit()        

        # if self.fn_type in ['F0_energy_untweaked', 'F0_energy_depth_up', 'F0_hamming']:
        #     self.current_len = self._len_move_to_left()
        #     self.current_gate_count = self._get_gate_count()
        #     self.current_cost = self._get_average_cost()
        #     self.current_degree = self._get_average_degree()
        
        hamming_distance, matched, stabilizers_save, total_gate_count, stim_circuit = self.get_hamming()
        stabilizers_save = [str(ps) for ps in stabilizers_save]

        if hamming_distance < self.min_hamming:
            self.min_hamming = hamming_distance
            print(self.original_stabilizers)
            print(stabilizers_save)
            print('MATCHED: ', matched)

        rwd = self.reward_fn(hamming_distance, matched, total_gate_count)
        self.rwd = rwd

        # if self.fn_type in ['F0_energy_untweaked', 'F0_energy_depth_up', 'F0_hamming', 'F0_matched']:
        #     self.max_len = max(self.max_len, self.current_len)
        #     self.max_cost = max(self.max_cost, self.current_cost)

        self.prev_hamming = np.copy(hamming_distance)

        self.error = hamming_distance
        self.error_noiseless = self.error

        energy_done = int(self.error <= self.done_threshold)

        layers_done = self.step_counter == (self.num_layers - 1)

        done = int(energy_done or layers_done)

        self.previous_action = copy.deepcopy(action)
        self.generators_save = 0

        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(self.current_bond_distance)] = copy.deepcopy(self.curriculum)
        
        next_state = next_state[:, :self.num_qubits]
        return next_state.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done

    def reset(self):
        """
        Returns randomly initialized state of environment.
        State is a torch Tensor of size (5 x number of layers)
        1st row [0, num of qubits-1] - denotes qubit with control gate in each layer
        2nd row [0, num of qubits-1] - denotes qubit with not gate in each layer
        3rd, 4th & 5th row - rotation qubit, rotation axis, angle
        !!! When some position in 1st or 3rd row has value 'num_qubits',
            then this means empty slot, gate does not exist (we do not
            append it in circuit creator)
        """
        
        state = torch.zeros((self.num_layers, self.num_qubits, self.num_qubits))
        self.state = state
        
        self.current_number_of_cnots = 0
        self.current_action = [self.num_qubits]*2
        self.illegal_actions = [[]]*self.num_qubits

        self.make_circuit(state)
        self.step_counter = -1
        
        self.moments = [0]*self.num_qubits

        self.current_bond_distance = self.geometry[-3:]
        self.curriculum = copy.deepcopy(self.curriculum_dict[str(self.current_bond_distance)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())

        self.prev_hamming, _, _, _, _ = self.get_hamming(state)

        state = state[:, :self.num_qubits]
        return state.reshape(-1).to(self.device)

    def make_circuit(self, thetas=None):
        """
        CHANGED TO RETURN STIM CIRCUIT

        based on the angle of first rotation gate we decide if any rotation at
        a given qubit is present i.e.
        if thetas[0, i] == 0 then there is no rotation gate on the Control quibt
        if thetas[1, i] == 0 then there is no rotation gate on the NOT quibt
        CNOT gate have priority over rotations when both will be present in the given slot
        """
        state = self.state.clone()
                
        stim_circuit = stim.Circuit()

        # adding a first layer of H gates
        for qub in range(self.num_qubits):
            stim_circuit.append_operation("H", [qub])
        
        for i in range(self.num_layers):
            cnot_pos = np.where(state[i][0:self.num_qubits] == 1)
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]
            
            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    # replace CNOT with CZ gate
                    # stim_circuit.append_operation("CNOT", [ctrl[r], targ[r]])
                    stim_circuit.append_operation("CZ", [ctrl[r], targ[r]])

        return stim_circuit
    
    def get_xyz_distance(self, pauli_a: str, pauli_b: str):
        pauli_a = str(pauli_a)
        pauli_b = str(pauli_b)

        n = len(pauli_a)

        mod_hamming = 0
        matched = 0
        # start from 1 to ignore sign
        for i in range(1, n):
            a = pauli_a[i]
            b = pauli_b[i]

            if a == b and a != '_':
                matched += 1

            if a == b:
                mod_hamming += 0
            elif a == '_' or b == '_':
                mod_hamming += self.xyz_val1
            else:
                mod_hamming += 1

        return mod_hamming / self.xyz_div1, matched

    
    def get_hamming_distance(self, current_stabilizers: list[str]):
        """
        Computes the Hamming distance between the original stabilizers of the surface code 
        and the stabilizers of the current circuit.
        """
        current_stabilizers = [str(ps) for ps in current_stabilizers]

        if len(current_stabilizers) == 0:
            return self.max_hamming, 0.0
        
        if len(self.original_stabilizers) != len(current_stabilizers):
            return self.max_hamming, 0.0
        
        hamming_sum = 0.0
        matched_sum = 0.0

        for x, y in zip(self.original_stabilizers, current_stabilizers):
            h, m = self.get_xyz_distance(x, y)
            hamming_sum += h
            matched_sum += m
        
        hamming_sum = float(hamming_sum)
        matched_sum = float(matched_sum)
        return hamming_sum, matched_sum


    def get_matched_count(self, current_stabilizers: list[str]) -> int:
        matched_count = 0
        current_stabilizers = [str(ps)[1:] for ps in current_stabilizers]
        original_stabilizers = [str(ps)[1:] for ps in self.original_stabilizers]

        for s in current_stabilizers:
            if s in original_stabilizers:
                matched_count += 1
        
        return float(matched_count)

    
    def qulacs_to_stim(self, qulacs_circuit) -> stim.Circuit:
        stim_circuit = stim.Circuit()

        total_gate_count = qulacs_circuit.get_gate_count()

        for i in range(total_gate_count):
            the_gate = qulacs_circuit.get_gate(i)
            name_of_gate = the_gate.get_name()

            if name_of_gate != 'CNOT':
                stim_circuit.append_operation("H", [the_gate.get_target_index_list()[0]])
            else:
                stim_circuit.append_operation("CX", [the_gate.get_control_index_list()[0], 
                                              the_gate.get_target_index_list()[0]])
        
        return stim_circuit, total_gate_count
    

    def get_hamming(self, thetas=None):
        stim_circuit = self.make_circuit(thetas)

        total_gate_count = 0

        s = stim.TableauSimulator()
        s.do_circuit(stim_circuit)
        current_canonical_stabilizers = s.canonical_stabilizers()

        hamming_distance, matched = self.get_hamming_distance(current_canonical_stabilizers)
        print(hamming_distance)

        if hamming_distance == 0.0:
            print(stim_circuit)
            print(current_canonical_stabilizers)
        
        matched = self.get_matched_count(current_canonical_stabilizers)

        return hamming_distance, matched, current_canonical_stabilizers, total_gate_count, stim_circuit

    
    def _get_average_cost(self) -> float:
        """
        Returns an estimation of the cost that the circuit has. This is computed as a weighted average of the gates
        in the circuit. It is assumed that multi-qubit gates are more expensive than single qubit gates.
        """
        div_by = self.CNOT_WEIGHT + self.H_WEIGHT
        cnots: int = 0
        hs: int = 0
        for moment in self.current_circuit:
            for op in moment:
                if op.gate == cirq.CNOT:
                    cnots += 1
                else:
                    hs += 1
        return (cnots * self.CNOT_WEIGHT + hs * self.H_WEIGHT) / div_by

    def _get_gate_count(self) -> int:
        counter: int = 0
        for moment in self.current_circuit:
            counter += len(moment)
        return counter

    def _get_average_degree(self) -> float:
            """
            This function computes the degree of CNOTs that each qubits has. The more parallel CNOTs (with multiple
            targets) in the circuit, the better (lower) the degree value should be.

            Returns:
            -------
                the average of all degrees
            """
            degrees = np.zeros(len(self.current_circuit.all_qubits()))
            qubit_dict = dict()
            for qubit in self.current_circuit.all_qubits():
                qubit_dict[qubit] = len(qubit_dict)

            for moment in self.current_circuit:
                for op in moment:
                    if op.gate == cirq.CNOT: 
                        indices = [qubit_dict[q] for q in op.qubits]
                        degrees[indices] += 1

            if np.mean(degrees) == 0:
                return 1.0
            return np.mean(degrees)

    def _len_move_to_left(self) -> int:
        n_circuit = cirq.Circuit(self.current_circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST)
        return len(n_circuit)
            

    def reward_fn(self, hamming_distance, matched=None, total_gate_count=None):
        """
        This returns the reward value that the agent receives based on the current circuit
        and the initial values that the circuit had.
        """
        if self.fn_type == 'F0_hamming':
            sham = hamming_distance / self.max_hamming
            e = np.exp((self.param_center - sham) ** 2 / 2 * (self.param_sigma ** 2))

            if sham > self.param_center:
                e = 1 - e
            else:
                e = e - 1

            # return 100 * e
            # return np.exp(-2 * hamming_distance)
            # return -hamming_distance
            # sham = hamming_distance / self.max_hamming
            # e = np.exp((self.param_center - sham) ** 2 / 2* (self.param_sigma**2))

            # if sham > self.param_center:
            #     e = 1 - e
            # else:
            #     e = e - 1

            factor = 100
            add = 0
            boost = self.param_center / 2
            if sham < boost:
                # factor += (boost/sham)
                add += self.boost

            return add + factor * e
            # return 1/hamming_distance

    def illegal_action_new(self):
        action = self.current_action
        illegal_action = self.illegal_actions
        ctrl, targ = action[0], (action[0] + action[1]) % self.num_qubits
        # rot_qubit, rot_axis = action[2], action[3]

        if ctrl < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        # if ill_ac[2] == self.num_qubits:
                        
                        #     if ctrl == ill_ac[0] or ctrl == ill_ac_targ:
                        #         illegal_action[ill_ac_no] = []
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break

                        #     elif targ == ill_ac[0] or targ == ill_ac_targ:
                        #         illegal_action[ill_ac_no] = []
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break
                            
                        #     else:
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break
                        # else:
                        #     if ctrl == ill_ac[2]:
                        #         illegal_action[ill_ac_no] = []
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break

                        #     elif targ == ill_ac[2]:
                        #         illegal_action[ill_ac_no] = []
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break
                        #     else:
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break                          
            else:
                illegal_action[0] = action

                            
        # if rot_qubit < self.num_qubits:
        #     are_you_empty = sum([sum(l) for l in illegal_action])
            
        #     if are_you_empty != 0:
        #         for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
        #             if len(ill_ac) != 0:
        #                 ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        # if ill_ac[0] == self.num_qubits:
                            
                        #     if rot_qubit == ill_ac[2] and rot_axis != ill_ac[3]:
                        #         illegal_action[ill_ac_no] = []
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break
                            
                        #     elif rot_qubit != ill_ac[2]:
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break
                        # else:
                        #     if rot_qubit == ill_ac[0]:
                        #         illegal_action[ill_ac_no] = []
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break
                                        
                        #     elif rot_qubit == ill_ac_targ:
                        #         illegal_action[ill_ac_no] = []
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break
                            
                        #     else:
                        #         for i in range(1, self.num_qubits):
                        #             if len(illegal_action[i]) == 0:
                        #                 illegal_action[i] = action
                        #                 break 
            # else:
            #     illegal_action[0] = action
        
        for indx in range(self.num_qubits):
            for jndx in range(indx+1, self.num_qubits):
                if illegal_action[indx] == illegal_action[jndx]:
                    if jndx != indx +1:
                        illegal_action[indx] = []
                    else:
                        illegal_action[jndx] = []
                    break
        
        for indx in range(self.num_qubits-1):
            if len(illegal_action[indx])==0:
                illegal_action[indx] = illegal_action[indx+1]
                illegal_action[indx+1] = []
        
        illegal_action_decode = []
        for key, contain in dictionary_of_cz_actions(self.num_qubits).items():
            for ill_action in illegal_action:
                if ill_action == contain:
                    illegal_action_decode.append(key)
        self.illegal_actions = illegal_action
        return illegal_action_decode


if __name__ == "__main__":
    pass
