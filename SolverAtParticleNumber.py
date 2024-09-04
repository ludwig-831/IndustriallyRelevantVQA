import numpy as np
from scipy.special import binom
from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import Statevector

from itertools import permutations

class SolverAtParticleNumber():

    def __init__(self,
                 operator_loader,
                 ):
        """
        Args:
            operator_loader (OperatorLoader): Internal operator loader
            num_qubits (int): Number of qubits, must be an even number
            num_s_up (int): Number of spin up particles
            num_s_down (int): Number of spin down particles
        """
        self.operator_loader = operator_loader
        self.full_matrix = self.operator_loader.penalized_pauli_hamiltonian_matrix
        self.num_qubits = self.operator_loader.molecule_data["num_qubits"]
        self.num_s_up = self.operator_loader.molecule_data["num_s_up"]
        self.num_s_down = self.operator_loader.molecule_data["num_s_down"]



        # size of spin sector
        self.sector_size = self.num_qubits//2

        self._validate_inputs()

        self.permuted_bitstrings = self.get_permuted_bitstrings()
        self.subspace_matrix = self.get_subspace_matrix()

        self.gse = None
        self.subspace_gs = None
        self.full_gs = None

        self.is_solved = False


    def _validate_inputs(self):
        assert self.num_qubits%2 == 0, f"Qubit number must be even, received {self.num_qubits}"
        assert self.full_matrix.shape[0] == 2**self.num_qubits
        assert (self.num_s_up <= self.sector_size) and (self.num_s_up <= self.sector_size)



    def get_subsector_permut_list(self, num_particles):
        vanilla_string = list(self.sector_size*"0") # list for item assignment
        for i in range(num_particles):
            vanilla_string[-(i+1)] = "1"

        # get permutations
        permut_list = []
        for permut in permutations(vanilla_string):
             permut_list.append("".join(permut))

        permut_list = list(set(permut_list))

        # sanity check that number of permutations equals binomial coefficient
        assert len(permut_list) == binom(self.sector_size, num_particles)

        permut_list.sort()

        return permut_list


    def get_permuted_bitstrings(self):
        """
        QISKIT ORDERING
        Returns all possible bit strings according to a given qubit number and
        number of spin-up and spin-down electrons.
        Ordered w.r.t. their numerical values.

        Returns:
            List[String]
        """

        s_up_permut_list = self.get_subsector_permut_list(self.num_s_up)
        s_down_permut_list = self.get_subsector_permut_list(self.num_s_down)

        # create list of all allowed basis states in the total system
        all_permutations_combined = []
        # interweave strings from sectors
        for up_string in s_up_permut_list:
            for down_string in s_down_permut_list:
                # empty string for combined bitstring
                string_combined = ""
                for i in range(self.sector_size):
                    string_combined = (down_string[-(i+1)]
                                     + up_string[-(i+1)]
                                     + string_combined)

                all_permutations_combined.append(string_combined)

        # sanity check: make sure every string appears exactly once
        assert len(set(all_permutations_combined)) == len(all_permutations_combined), (
            "Some bitstrings appear more than once, this should not be the case"
        )

        all_permutations_combined.sort()

        return all_permutations_combined

    def get_subspace_indices(self):
        """
        Determines the indices needed to create the reduced matrix that respects
        all symmetries.

        Returns:
            List[Int]
        """

        index_list = []
        for bitstring in self.permuted_bitstrings:
            index = int(bitstring, base=2)
            index_list.append(index)

        return index_list

    def get_subspace_matrix(self):
        dim = len(self.permuted_bitstrings)
        subspace_matrix = np.zeros((dim, dim), dtype=np.float64)
        self.subspace_indices = self.get_subspace_indices()

        # self.permuted_bitstrings = ["0001", "0100"]
        # -> index of string is index in reduced matrix, string in int form is
        # index in full hamiltonian matrix
        for subspace_row_index, full_row_index in enumerate(self.subspace_indices):
            for subspace_col_index, full_col_index in enumerate(self.subspace_indices):
                matrix_element = self.full_matrix[full_row_index, full_col_index]
                assert np.isclose(np.imag(matrix_element), 0), "Full Hamiltonian matrix has non-zero complex values"
                subspace_matrix[subspace_row_index, subspace_col_index] = np.real(matrix_element)

        return subspace_matrix

    def _solve_subspace(self, k):
        print("subspace matrix", self.subspace_matrix.shape)
        if self.subspace_matrix.shape == (1,1):
            eigvals, eigvecs = eigsh(self.subspace_matrix, k=1, which="SA")
            self.gse = eigvals[0]
            self.first_excited_energy = None
            self.energy_gap = None
            self.subspace_gs = eigvecs[:,0]
            return eigvals, eigvecs

        eigvals, eigvecs = eigsh(self.subspace_matrix, k=k, which="SA")
        # if the spin sectors are fully occupied or empty there will only be one eigval
        if len(eigvals) > 1:
            assert eigvals[0] != eigvals[1], (f"Ground state energy of subspace matrix is degenerate: {eigvals}")

        self.gse = eigvals[0]

        assert np.isclose(self.gse, self.operator_loader.molecule_data["gse"]), (
            "Hamiltonian does not have correct energy. If Hamiltonian is penalized, try increasing the penalty scaling.",
            self.gse, self.gse - self.operator_loader.molecule_data["gse"]
        )
        print("Hamiltonian has correct energy.")

        self.first_excited_energy = eigvals[1]
        self.subspace_gs = eigvecs[:,0]

        self.energy_gap = self.first_excited_energy - self.gse


        print("Energy Gap:", self.energy_gap)

        return eigvals, eigvecs

    def _expand_subspace_vec(self):
        assert self.subspace_gs is not None, "Must solve subspace first."
        vec_shape = (self.full_matrix.shape[0],)
        full_vec = np.zeros(vec_shape)
        for subspace_index, full_index in enumerate(self.subspace_indices):
            element = self.subspace_gs[subspace_index]
            full_vec[full_index] = element

        self.full_gs = full_vec

        # save largest coefficientis_solved
        self.largest_ci_sq_index, self.largest_ci_sq = np.argmax(np.abs(full_vec)**2), np.max(np.abs(full_vec)**2)
        self.largest_ci_sq_binary_index = self.int_to_n_bit_binary(self.largest_ci_sq_index)


    def solve(self, k=2):
        self._solve_subspace(k=k)
        self._expand_subspace_vec()

        s2_expectation_val = np.real(
            Statevector(self.full_gs).expectation_value(self.operator_loader.spin_squared_op_jw)
            )

        assert np.isclose(s2_expectation_val, self.operator_loader.molecule_data["s_squared"]), (
            f"Spin-squared expectation value of penalized Hamiltonian does not have the correct value: {s2_expectation_val}"
            + f'. Expected: {self.operator_loader.molecule_data["s_squared"]}. Try increasing the penalty term.'
        )

        self.is_solved = True

        return self.gse, self.full_gs

    def get_hartree_string(self):
        spin_up_sector_string = (self.sector_size-self.num_s_up)*"0" + self.num_s_up*"1"
        spin_down_sector_string = (self.sector_size-self.num_s_down)*"0" + self.num_s_down*"1"

        hartree_string = ""
        for i in range(self.sector_size):
            hartree_string = hartree_string + spin_down_sector_string[i] + spin_up_sector_string[i]

        return hartree_string

    def get_hartree_vec(self):
        hartree_string = self.get_hartree_string()
        return self.get_vec_from_bitstring(hartree_string)


    def get_vec_from_bitstring(self, bitstring):
        vec_shape = (2**self.num_qubits,)
        vec = np.zeros(vec_shape)

        index = int(bitstring, base=2)
        vec[index] = 1

        return vec


    def get_hartree_energy(self):
        # make sure self.full_matrix exists
        assert self.is_solved

        hartree_vec = Statevector(self.get_hartree_vec())
        hartree_energy = np.real(hartree_vec.expectation_value(self.full_matrix))

        return hartree_energy


    def int_to_n_bit_binary(self, integer) -> str:
        format_string = "{:0"+f"{self.num_qubits}"+"b}"
        n_bit_binary = format_string.format(integer)
        return n_bit_binary
