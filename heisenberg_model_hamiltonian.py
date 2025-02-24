import netket as nk
import numpy as np
import scipy.linalg as la

# Define the 3x3 lattice
lattice = 3
graph = nk.graph.Square(lattice)
# nk.graph.Square()
# Define the Hilbert space (spin-1/2 particles)
hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)

# Create the Heisenberg Hamiltonian
hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=graph, J=0.25)

# Print the Hamiltonian
print(hamiltonian)

# If you want to get the matrix representation of the Hamiltonian
matrix = hamiltonian.to_dense()
print("Hamiltonian matrix:")
print(matrix)

# Compute the ground state energy using exact diagonalization
gs_energy = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=False)
print(f"Ground state energy: {gs_energy}")


complete_dict = dict()
hamiltonian_mat = matrix
print(hamiltonian_mat.shape)
mol = 'heisenberg'
qubits = lattice**2
lattice_structure = '3x3'

complete_dict['hamiltonian'] = hamiltonian_mat
complete_dict['eigvals'] = la.eig(hamiltonian_mat)[0].real
complete_dict['weights'] = 0
complete_dict['paulis'] = 0
complete_dict['energy_shift'] = 0

np.savez(f'mol_data/{mol}_{qubits}q_{lattice_structure}', **complete_dict)