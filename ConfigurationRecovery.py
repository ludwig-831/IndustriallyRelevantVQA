from typing import List, Dict
from OperatorLoader import OperatorLoader
import numpy as np
from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import Statevector
import random
from copy import deepcopy

class ConfigurationRecovery:
    def __init__(self, measurement_dict: List[Dict], K: int, operator_loader):
        self.measurement_dict = measurement_dict
        self.total_samples = round(np.sum([x for x in measurement_dict.values()]))
        assert self.total_samples == 10000
        self.K = K # number of batches
        self.operator_loader = operator_loader
        self.num_s_up = operator_loader.molecule_data["num_s_up"]
        self.num_s_down = operator_loader.molecule_data["num_s_down"]
        self.num_qubits = operator_loader.molecule_data["num_qubits"]

        # get correct and faulty bitstrings in separate lists of dicts
        self.correct_bitstrings, self.faulty_bitstrings = self.categorize_dict(self.measurement_dict)
        assert self.samples_conserved()


        self.setup = False



    def categorize_dict(self, bitstring_dict: Dict):
        correct_bitstrings = {}
        faulty_bitstrings = {}
        for bitstring, count in bitstring_dict.items():
            # convert bitstring to list of 1s and 0s
            bitstring_list = [int(x) for x in bitstring]
            # getself.faulty_bitstrings.keys() measured particle numbers
            meas_n_up = sum(bitstring_list[1::2])
            meas_n_down = sum(bitstring_list[::2])

            # check for measurements with correct particle numbers
            if meas_n_up == self.num_s_up and meas_n_down == self.num_s_down:
                correct_bitstrings[bitstring] = count
            else:
                faulty_bitstrings[bitstring] = count

        return correct_bitstrings, faulty_bitstrings

    def run(self):
        while len(self.faulty_bitstrings) > 0:
            self.run_setup()
            self.run_selfconsistent_iteration()

        return self.correct_bitstrings

    def run_setup(self):

        self.correct_batches = self.split_into_batches(self.correct_bitstrings, self.K)
        self.setup = True

        groundstates = []
        for batch in self.correct_batches:
            groundstates.append(self.get_groundstate(batch))

        self.groundstates = groundstates
        self.occupation_vector = self.get_occupation_vector()

        assert self.samples_conserved()


    def run_selfconsistent_iteration(self):
        assert self.setup == True, "Perform ConfigurationRecovery.run_setup() first"

        faulty_keys = self.faulty_bitstrings.keys()

        corrected_faulty_dict = {}

        for faulty_bitstring in faulty_keys:
            # convert bitstring to list of 1s and 0s
            bitstring_list = [int(x) for x in faulty_bitstring]
            meas_up_list = bitstring_list[1::2]
            meas_down_list = bitstring_list[::2]
            # get measured particle numbers
            meas_n_up = sum(meas_up_list)
            meas_n_down = sum(meas_down_list)
            # get indices of qubits that could potentially be flipped
            pickable_indices_up, pickable_indices_down = self.get_pickable_indices(
                meas_up_list, meas_down_list, meas_n_up, meas_n_down,
            )
            # get number of qubits that needs to be flipped
            number_flip_up = round(np.abs(meas_n_up - self.num_s_up))
            number_flip_down = round(np.abs(meas_n_down - self.num_s_down))
            # get qubits that will be flipped if successful in probabilistic trial below
            flip_indices_up = random.sample(pickable_indices_up, number_flip_up)
            flip_indices_down = random.sample(pickable_indices_down, number_flip_down)
            # there should only be unique elements in the lists
            assert len(flip_indices_up) == len(set(flip_indices_up))
            assert len(flip_indices_down) == len(set(flip_indices_down))

            # concatenate lists
            flip_indices_combined = flip_indices_up + flip_indices_down

            # compare occupation of bitstring and occupation vector
            # pickable indices indicate the indices of qubits, i.e. index 9 corresponds to qubit 0
            for index in flip_indices_combined:
                x = bitstring_list[index]
                assert x == 1 or x == 0
                n_vec = self.occupation_vector # occ vec is updated in setup between iterations
                n = n_vec[index]

                diff = np.abs(x - n)
                # probability of flipping
                w = self.relu(diff)
                if self.flip_bit(w):
                    # flip bit in bitstring list
                    flipped_bit = (bitstring_list[index] + 1)%2
                    assert flipped_bit == 1 or flipped_bit == 0
                    bitstring_list[index] = flipped_bit

            bitstring_str_list = [str(x) for x in bitstring_list]
            corrected_bitstring = "".join(bitstring_str_list)

            # dict now contains correct and faulty bitstrings -> categorize and re-distribute
            if corrected_bitstring in corrected_faulty_dict.keys():
                corrected_faulty_dict[corrected_bitstring] += self.faulty_bitstrings[faulty_bitstring]
            else:
                corrected_faulty_dict[corrected_bitstring] = self.faulty_bitstrings[faulty_bitstring]

        temp_correct_dict, temp_faulty_dict = self.categorize_dict(corrected_faulty_dict)

        sum1 = np.sum([x for x in temp_correct_dict.values()])
        sum2 = np.sum([x for x in temp_faulty_dict.values()])
        sum3 = np.sum([x for x in corrected_faulty_dict.values()])

        assert round(sum1 + sum2) == round(sum3), "Error in Categorize dict"

        self.faulty_bitstrings = temp_faulty_dict

        for key, val in temp_correct_dict.items():
            if key in self.correct_bitstrings:
                self.correct_bitstrings[key] += val
            else:
                self.correct_bitstrings[key] = val



            # set self.groundstates, self.correct_bitstrings, self.faulty_bitstrings again


    def samples_conserved(self):
        correct = np.sum([x for x in self.correct_bitstrings.values()])
        faulty = np.sum([x for x in self.faulty_bitstrings.values()])
        assert round(correct + faulty) == self.total_samples

        return True

    def flip_bit(self, w):
        if np.random.rand() <= w:
            return True
        else:
            return False

    def relu(self, y):
        delta = 0.2
        # h = (self.num_s_up + self.num_s_down)/self.num_qubits
        h = 0.8
        if y <= h:
            return delta*y/h
        else:
            return delta + (1 - delta)*(y-h)/(1-h)


    def get_pickable_indices(self, meas_up_list, meas_down_list,
            meas_n_up, meas_n_down):
        """
        Determines the indices in the subsectors that can be picked to be flipped.
        """
        pickable_indices_up = []
        pickable_indices_down = []


        if meas_n_up > self.num_s_up:
            # determine indices to be sampled from
            for index, bit in enumerate(meas_up_list):
                if bit == 1:
                    pickable_indices_up.append(index*2 + 1)
        elif meas_n_up < self.num_s_up:
            for index, bit in enumerate(meas_up_list):
                if bit == 0:
                    pickable_indices_up.append(index*2 + 1)

        if meas_n_down > self.num_s_down:
            # determine indices to be sampled from
            for index, bit in enumerate(meas_down_list):
                if bit == 1:
                    pickable_indices_down.append(index*2)
        elif meas_n_down < self.num_s_down:
            for index, bit in enumerate(meas_down_list):
                if bit == 0:
                    pickable_indices_down.append(index*2)

        return pickable_indices_up, pickable_indices_down


    def split_into_batches(self, original_dict, num_batches):
        """
        Helper function to split a set of measurements into K batches
        """
        # Initialize the list of dictionaries for each batch
        batches = [{} for _ in range(num_batches)]

        # Create a list to keep track of total counts in each batch
        counts = [0] * num_batches

        # Shuffle the keys to ensure random distribution
        keys = list(original_dict.keys())
        random.shuffle(keys)

        for key in keys:
            count = original_dict[key]
            while count > 0:
                # Find the batch with the current minimum count
                min_batch_index = counts.index(min(counts))

                # Determine the portion to distribute to the selected batch
                portion = min(count, 1)

                # Add the key to the selected batch
                if key in batches[min_batch_index]:
                    batches[min_batch_index][key] += portion
                else:
                    batches[min_batch_index][key] = portion

                # Update the counts for the selected batch
                counts[min_batch_index] += portion
                count -= portion

        return batches

    def merge_batches(self, batches):
        merged_dict = {}

        for batch in batches:
            for key, val in batch.items():
                if key in merged_dict:
                    merged_dict[key] += val
                else:
                    merged_dict[key] = val

        return merged_dict



    def get_occupation_vector(self):
        if self.setup == False:
            self.run_setup()
        groundstates = self.groundstates
        K = self.K
        occupation_vector = [None for _ in range(self.operator_loader.num_qubits)]

        for qubit_index in range(self.operator_loader.num_qubits):
            sum_exp_vals = 0
            for k in range(K):
                state = groundstates[k]
                number_op_jw = self.operator_loader.get_number_op(qubit_index)
                exp_val = np.real(state.expectation_value(number_op_jw))
                sum_exp_vals += exp_val

            normalized_exp_val = sum_exp_vals/K

            occupation_vector[qubit_index] = normalized_exp_val

        # return occupation vector in qiskit ordering, [qubit9, qubit8, ..., qubit0]
        return occupation_vector[::-1]


    def get_groundstate(self, batch):
        """
        batch = {"1100100": 3, "111100": 4, etc.}

        get groundstate from reduced hamiltonian in subspace of batch
        """
        p_z_measured = np.zeros(2 ** self.operator_loader.num_qubits)
        for bitstring, value in batch.items():
            index = int(bitstring, 2)
            p_z_measured[index] = value
        p_z_measured = p_z_measured / np.sum(p_z_measured)

        mask = p_z_measured > 0 # True for all measured bitstrings
        red_hamiltonian = self.operator_loader.penalized_pauli_hamiltonian_matrix[mask][:,mask]
        eigvals, eigvecs = eigsh(red_hamiltonian, k=2, which='SA')
        gs_ind = np.argmin(eigvals)
        red_gs_energy = eigvals[gs_ind]
        red_ground_state = eigvecs[:,gs_ind]

        # get reduced ground state in full Hilbert space
        red_ground_state_full = np.zeros(2**self.operator_loader.num_qubits, dtype=np.complex128)
        red_ground_state_full[mask] = red_ground_state
        red_gs_full_statevector = Statevector(red_ground_state_full)

        return red_gs_full_statevector


    def get_groundstate_energy(self, batch):
        """
        batch = {"1100100": 3, "111100": 4, etc.}

        get groundstate energy from reduced hamiltonian in subspace of batch
        """
        p_z_measured = np.zeros(2 ** self.operator_loader.num_qubits)
        for bitstring, value in batch.items():
            index = int(bitstring, 2)
            p_z_measured[index] = value
        p_z_measured = p_z_measured / np.sum(p_z_measured)

        mask = p_z_measured > 0 # True for all measured bitstrings
        red_hamiltonian = self.operator_loader.penalized_pauli_hamiltonian_matrix[mask][:,mask]
        eigvals, eigvecs = eigsh(red_hamiltonian, k=2, which='SA')
        gs_ind = np.argmin(eigvals)
        red_gs_energy = eigvals[gs_ind]

        return red_gs_energy
