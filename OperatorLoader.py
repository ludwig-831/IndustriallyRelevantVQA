import numpy as np
import json

from qiskit_nature.second_q.operators import FermionicOp, PolynomialTensor
from qiskit_nature.second_q.mappers import JordanWignerMapper

from qiskit.quantum_info import Pauli, SparsePauliOp

import qiskit_nature
qiskit_nature.settings.use_pauli_sum_op = False


class OperatorLoader():

    def __init__(
        self,
        molecule_name: str,
        path_to_molecules_folder: str = "./molecules/"
    ):

        self.molecule_name = molecule_name
        self.path_to_folder = path_to_molecules_folder + f"{self.molecule_name}/"

        with open(self.path_to_folder + "data_file.json", "r") as read_file:
            self.molecule_data = json.load(read_file)

        self.num_qubits = self.molecule_data["num_qubits"]
        assert self.num_qubits%2 == 0

        self.desired_s2 = self.molecule_data["s_squared"]

        # choose a mapper, for my ansatz i need to measure a JW Hamiltonian in the end
        # (or swap my qubits to be in a new basis first)
        self.mapper = JordanWignerMapper()

        self.total_number_op_f = None
        self.total_number_op_jw = None
        self.load_total_number_op()

        self.spin_up_number_op_f = None
        self.spin_up_number_op_jw = None
        self.load_spin_up_number_op()

        self.spin_down_number_op_f = None
        self.spin_down_number_op_jw = None
        self.load_spin_down_number_op()

        self.spin_squared_op_f = None
        self.spin_squared_op_jw = None
        self.load_spin_squared_op()

        self.fermionic_hamiltonian = None
        self.pauli_hamiltonian = None
        self.pauli_hamiltonian_matrix = None
        self.penalized_pauli_hamiltonian = None
        self.penalized_pauli_hamiltonian_matrix = None

        self.load_hamiltonian()


    def load_total_number_op(self):
        number_op = 0
        for i in range(self.num_qubits):
            number_op += FermionicOp(
                {f"+_{i} -_{i}": 1.0},
                num_spin_orbitals=self.num_qubits
                )
        self.total_number_op_f = number_op
        self.total_number_op_jw = self.mapper.map(number_op)

    def load_spin_up_number_op(self):
        # QISKIT ORDERING!
        spin_up_number_op = 0
        for i in range(0, self.num_qubits, 2):
            spin_up_number_op += FermionicOp(
                {f"+_{i} -_{i}": 1.0},
                num_spin_orbitals=self.num_qubits
                )
        self.spin_up_number_op_f = spin_up_number_op
        self.spin_up_number_op_jw = self.mapper.map(spin_up_number_op)


    def load_spin_down_number_op(self):
        # QISKIT ORDERING!
        spin_down_number_op = 0
        for i in range(1, self.num_qubits, 2):
            spin_down_number_op += FermionicOp(
                {f"+_{i} -_{i}": 1.0},
                num_spin_orbitals=self.num_qubits
                )
        self.spin_down_number_op_f = spin_down_number_op
        self.spin_down_number_op_jw = self.mapper.map(spin_down_number_op)

    def get_number_op(self, qubit_index):
        number_op = FermionicOp(
            {f"+_{qubit_index} -_{qubit_index}": 1.0},
            num_spin_orbitals=self.num_qubits
            )

        number_op_jw = self.mapper.map(number_op)

        return number_op_jw



    def load_spin_squared_op(self):
        """
        Implements the spin squared operator
        (page 4: https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3/pdf)

        S^2 = sum_{pq} (p↑ q↑^+) (p↓^+ q↓) + (N↑ - N↓)/2 + (N↑ - N↓)^2/4
        = S-S+ + Sz + Sz^2
        """
        sum_4local = 0
        for p in range(0, self.num_qubits//2):
            for q in range(0, self.num_qubits//2):
                sum_4local += FermionicOp(
                    {f"-_{2*p} +_{2*q} +_{2*p+1} -_{2*q+1}": 1.0},
                    num_spin_orbitals=self.num_qubits
                )

        sz = (self.spin_up_number_op_f - self.spin_down_number_op_f)/2
        sz2 = sz.compose(sz)

        spin_squared_op = sum_4local + sz + sz2

        self.spin_squared_op_f = spin_squared_op
        spin_squared_op_jw = self.mapper.map(spin_squared_op)
        self.spin_squared_op_jw = spin_squared_op_jw.simplify()


    def load_hamiltonian(self):
        """
        Loads the Hamiltonian and saves it to the instance as a FermionicOp,
        a PauliSum and a scipy.sparse matrix.
        """

        self.reduced_matrix = np.load(self.path_to_folder + "hamiltonian_matrix.npy")

        # load tensors
        self.core_energy = float(np.load(self.path_to_folder + "core_energy.npy"))
        self.one_body_tensor = np.load(self.path_to_folder + "one_body.npy")
        self.two_body_tensor = np.load(self.path_to_folder + "two_body.npy")

        assert np.isclose(self.core_energy, self.molecule_data["core_energy"]), (
            "Core energy in tensors is not the same as core energy provided by Lukas in the info.txt file"
            )

        # create dict with tensors for qiskit's PolynomialTensor
        data = {}
        data[""] = self.core_energy
        data["+-"] = self.one_body_tensor
        data["++--"] = self.two_body_tensor
        tensor = PolynomialTensor(data)
        # create FermionicOp from PolynomialTensor
        self.fermionic_hamiltonian = FermionicOp.from_polynomial_tensor(tensor)

        id_string = self.num_qubits*"I"
        id_op = Pauli(id_string)

        self.penalty_term = (self.spin_squared_op_jw - SparsePauliOp(id_op, self.desired_s2))
        self.penalty_term_sq = self.penalty_term.compose(self.penalty_term)

        # map FermionicOp to PauliSumOp via JordanWignerMapper
        self.pauli_hamiltonian = self.mapper.map(self.fermionic_hamiltonian)
        self.pauli_hamiltonian = self.pauli_hamiltonian.simplify()


        # Penalized Hamiltonian to enforce spin
        self.penalized_pauli_hamiltonian = self.pauli_hamiltonian + 1e-2*self.penalty_term_sq
        self.penalized_pauli_hamiltonian = self.penalized_pauli_hamiltonian.simplify()
        # get matrix form of Jordan-Wignered Hamiltonian
        self.pauli_hamiltonian_matrix = self.pauli_hamiltonian.to_matrix()
        self.penalized_pauli_hamiltonian_matrix = self.penalized_pauli_hamiltonian.to_matrix()
        # sanity check dimension of the Hamiltonian matrix
        assert 2**(self.num_qubits) == self.pauli_hamiltonian_matrix.shape[0]
