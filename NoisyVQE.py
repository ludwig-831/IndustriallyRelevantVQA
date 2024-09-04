from typing import List
import os
from copy import deepcopy
import pickle

import matplotlib.pyplot as plt

from scipy.optimize import minimize, OptimizeResult
import numpy as np

from qiskit import QuantumCircuit, transpile, qpy
from qiskit.quantum_info import Statevector
import sys
sys.path.append("../")
from Ansatz import Ansatz
from ShadowGrouping import Measurement_scheme, Zstring_Estimator
from qiskit_aer.primitives import Sampler

class NoisyVQE():

    def __init__(
        self,
        measurement_config: dict,
        backend_options: dict,
        ansatz: Ansatz,
        molecule_name: str,
        full_solver,
        operator_loader,
        ):

        # train on energy expectation value, not infidelity
        self.objective = "energy"

        self.measurement_config = measurement_config
        self.backend_options = backend_options
        self.ansatz = ansatz
        self.circuit = ansatz.circuit
        self.num_qubits = self.ansatz.num_qubits
        self.molecule_name = molecule_name
        self.full_solver = full_solver
        self.operator_loader = operator_loader

        # save all bitstrings to a dict
        self.samples_of_all_runs = {} # key: fev, val: postselected_outcomes
        self.samples_final_state = None

        self.molecule_data = operator_loader.molecule_data

        self.prepare_measurement_scheme(Nshots=self.measurement_config["Nshots"])

        # solve subspace for GSE, target state, etc.
        if not self.full_solver.is_solved:
            self.full_solver.solve()

        # ground state obtained from penalized hamiltonian
        self.target_state = self.full_solver.full_gs

        self.chemical_accuracy = 4/2625.49963948 # 4 kJ/mol in hartree


    def prepare_measurement_scheme(self,Nshots):
        # input penalized hamiltonian (!)
        self.method = Measurement_scheme(self.operator_loader.penalized_pauli_hamiltonian)
        self.estimator = Zstring_Estimator(self.method)

        self.estimator.propose_next_settings(Nshots)


    def optimize(self,
        maxfev: int,
        x0: list,
        ) -> OptimizeResult:
        """
        Optimize (minimize) infidelity

        Args:
            maxfev (int): Maximum number of function evaluations
            x0 (list): Set of initial variational parameters. Defaults to zero.
        Returns:
            OptimizeResult
        """

        # init persistence
        self.infidelity_descent = np.array([]) # saves state infidelity
        self.energy_descent = np.array([]) # tracks energy expectation value
        self.fev = 0 # function evaluations (= parameter updates)

        self.maxfev = maxfev

        if self.ansatz.num_layers == 0:
            self.save_results()
            return None

        if x0 == None:
            x0 = [0 for _ in range(self.circuit.num_parameters)]

        # not sure if setting bounds works
        bounds = [(-2*np.pi, 2*np.pi) for _ in range(self.circuit.num_parameters)]
        options = {"maxfev": maxfev, "disp": True}
        self.optimizer_res = minimize(
            fun=self.get_objective,
            x0=x0,
            method=nakanishi_fujii_todo, # source code below
            options=options,
            bounds=bounds,
            )

        print("Parameter updates:", self.fev-1, "Infidelity:", round(np.min(self.infidelity_descent), 6), "Current min:", round(np.min(self.energy_descent),6))

        # do sampling with optimized parameters
        self.sample_final_state()

        self.save_results()
        self.plot_result()

        return self.optimizer_res

    def get_objective(self,
        variational_params: List[float]
        ) -> float:
        """
        Calculates the infidelity of target state and circuit output depending
        on the variational parameters.

        Args:
            params (List[float]): List of variational parameters
            disp (bool): print current function eval count and infidelity
        Returns:
            float
        """

        # assert correct number of parameters
        assert self.circuit.num_parameters == len(variational_params)

        # bind parameters
        circ = self.circuit.bind_parameters(variational_params)

        # get statevector for infidelity
        circuit_statevec = Statevector(circ)

        # infidelity tracks current params in perfect circuit
        overlap = np.vdot(self.target_state, circuit_statevec.data)
        infidelity = 1 - np.abs(overlap)**2
        self.infidelity_descent = np.append(self.infidelity_descent, infidelity)


        # sampler executes circuit and yields sampled bitstrings
        sampler = Sampler(backend_options=self.backend_options,
                transpile_options={"optimization_level":3},
                skip_transpilation=False,
                )

        # estimator configured to measure penalized pauli hailtonian
        self.estimator.clear_outcomes() # clear previous outcomes
        outcomes = {}

        # loop through measurement settings
        for setting, dicts in self.estimator.settings_buffer.items():
            circuit = deepcopy(circ)
            circuit.compose(dicts["circuit"],inplace=True) # adds the measurement

            shots = int(dicts["nshots"])
            job = sampler.run(circuit, shots=shots,)
            results = job.result()

            # measured bitstrings are returned as base 10 numbers
            outcome_dict_base10 = results.quasi_dists[0]

            # convert base 10 numbers to bitstrings
            outcome_dict_bitstrings = {}
            for key in outcome_dict_base10:
                format_string = "{0:0" + str(self.num_qubits) + "b}"
                bit_string = format_string.format(key)

                count = round(outcome_dict_base10[key]*shots)

                outcome_dict_bitstrings[bit_string] = count

            # the energy estimator only requires this dictionary. All data conversion happens inside of Energy_Estimator
            outcomes[setting] = outcome_dict_bitstrings

            # measured bitstrings contain erroneous runs -> Postselection
            postselected_outcomes = {}
            for bitstring, count in outcome_dict_bitstrings.items():
                # convert bitstring to list of 1s and 0s
                bitstring_list = [int(x) for x in bitstring]
                # get measured particle numbers
                meas_n_up = sum(bitstring_list[1::2])
                meas_n_down = sum(bitstring_list[::2])

                # only track measurements with correct particle numbers
                if meas_n_up == self.molecule_data["num_s_up"] and meas_n_down == self.molecule_data["num_s_down"]:
                    postselected_outcomes[bitstring] = count

        # create a new estimator for the post-selected bitstrings
        postselection_estimator = Zstring_Estimator(self.method)
        virtual_Nshots = sum(postselected_outcomes.values())
        postselection_estimator.propose_next_settings(virtual_Nshots)
        postselection_estimator.get_outcomes({self.num_qubits*"Z": postselected_outcomes})

        # save samples to dict
        self.samples_of_all_runs[self.fev] = postselected_outcomes

        # post-selected expectation value of penalized hamiltonian
        energy = postselection_estimator.get_energy()

        energy_diff = energy - self.molecule_data["gse"]

        self.energy_descent = np.append(self.energy_descent, energy_diff)

        if self.fev%10 == 0:
            print("Parameter updates:", self.fev, "Infidelity:", round(infidelity, 6), "Current min:", round(np.min(self.energy_descent),6), end="\r", flush=True)

        self.fev += 1

        return energy_diff


    def sample_final_state(self):
        Nshots = self.measurement_config["Nshots_final_state"]
        self.prepare_measurement_scheme(Nshots=Nshots)

        # do the same as in get_objective
        # bind optimized parameters
        circ = self.circuit.bind_parameters(self.optimizer_res.x)

        # get statevector for infidelity
        circuit_statevec = Statevector(circ)


        # sampler executes circuit and yields sampled bitstrings
        sampler = Sampler(backend_options=self.backend_options,
                transpile_options={"optimization_level":3},
                skip_transpilation=False,
                )

        # estimator configured to measure penalized pauli hailtonian
        self.estimator.clear_outcomes() # clear previous outcomes
        outcomes = {}

        # loop through measurement settings
        for setting, dicts in self.estimator.settings_buffer.items():
            circuit = deepcopy(circ)
            circuit.compose(dicts["circuit"],inplace=True) # adds the measurement

            shots = int(dicts["nshots"])
            job = sampler.run(circuit, shots=shots,)
            results = job.result()

            # measured bitstrings are returned as base 10 numbers
            outcome_dict_base10 = results.quasi_dists[0]

            # convert base 10 numbers to bitstrings
            outcome_dict_bitstrings = {}
            for key in outcome_dict_base10:
                format_string = "{0:0" + str(self.num_qubits) + "b}"
                bit_string = format_string.format(key)

                count = round(outcome_dict_base10[key]*shots)

                outcome_dict_bitstrings[bit_string] = count

            # the energy estimator only requires this dictionary. All data conversion happens inside of Energy_Estimator
            outcomes[setting] = outcome_dict_bitstrings

            # measured bitstrings contain erroneous runs -> Postselection
            postselected_outcomes = {}
            for bitstring, count in outcome_dict_bitstrings.items():
                # convert bitstring to list of 1s and 0s
                bitstring_list = [int(x) for x in bitstring]
                # get measured particle numbers
                meas_n_up = sum(bitstring_list[1::2])
                meas_n_down = sum(bitstring_list[::2])

                # only track measurements with correct particle numbers
                if meas_n_up == self.molecule_data["num_s_up"] and meas_n_down == self.molecule_data["num_s_down"]:
                    postselected_outcomes[bitstring] = count

        # create a new estimator for the post-selected bitstrings
        postselection_estimator = Zstring_Estimator(self.method)
        virtual_Nshots = sum(postselected_outcomes.values())
        postselection_estimator.propose_next_settings(virtual_Nshots)
        postselection_estimator.get_outcomes({self.num_qubits*"Z": postselected_outcomes})

        # save samples to dict
        self.samples_final_state = postselected_outcomes

        # post-selected expectation value of penalized hamiltonian
        energy = postselection_estimator.get_energy()

        self.final_sample_energy_diff = energy - self.molecule_data["gse"]

        print("Final sample energy diff:", self.final_sample_energy_diff)
        print("Final sample Nshots", Nshots)




    def save_results(self):
        """
        Saves evolution of infidelity descent and best set of parameters.
        """

        self.subfolder_prefix = (f"./molecules/{self.molecule_name}"
                + f"/entanglement={self.ansatz.entanglement}/"
                + f"/l={self.ansatz.num_layers}/")


        if not os.path.isdir(self.subfolder_prefix):
            os.makedirs(self.subfolder_prefix)


        # save samples of all runs:

        with open(self.subfolder_prefix + 'samples_of_all_runs.pickle', 'wb') as f:
            pickle.dump(self.samples_of_all_runs, f)
        with open(self.subfolder_prefix + 'samples_final_state.pickle', 'wb') as f:
            pickle.dump(self.samples_final_state, f)

        # save optimization and optimal parameters
        np.savetxt(self.subfolder_prefix + "infidelity_descent.txt", self.infidelity_descent)
        np.savetxt(self.subfolder_prefix + "energy_descent.txt", self.energy_descent)
        if self.ansatz.num_layers > 0:
            np.savetxt(self.subfolder_prefix + "optimal_point.txt", self.optimizer_res.x)

        # save VQE settings
        file = open(self.subfolder_prefix + "vqe_info.txt", "w")

        file.write(f"Molecule: {self.molecule_name}\n")
        file.write(f"Number of Qubits: {self.ansatz.num_qubits}\n")
        file.write(f"Spin up: {self.ansatz.num_s_up}\n")
        file.write(f"Spin down: {self.ansatz.num_s_down}\n")

        file.write(f"Number of layers: {self.ansatz.num_layers}\n")
        file.write(f"Number of parameters: {self.ansatz.num_parameters}\n")

        if self.ansatz.num_layers > 0:
            file.write(f"Lowest infidelity: {self.infidelity_descent[-1]}\n")

            # get energy corresponding to lowest fidelity
            optimal_params = self.optimizer_res.x
            opt_circ = self.circuit.bind_parameters(optimal_params)
            circuit_statevec = Statevector(opt_circ)
        else:
            opt_circ = self.circuit
            circuit_statevec = Statevector(self.circuit)

        self.final_energy = np.real(circuit_statevec.expectation_value(self.operator_loader.pauli_hamiltonian_matrix))


        # saving the circuit
        with open(self.subfolder_prefix + 'original_circ.qpy', 'wb') as fd:
            qpy.dump(opt_circ.decompose(reps=3), fd)

        # saving the circuit
        with open(self.subfolder_prefix + 'original_parameterized_circ.qpy', 'wb') as fd:
            qpy.dump(self.circuit.decompose(reps=3), fd)

        file.write(f"Lowest energy: {self.final_energy}\n")
        file.write(f"Exact energy: {self.full_solver.gse}\n")
        file.write(f"First excited state energy: {self.full_solver.first_excited_energy}\n")

        # gse from penalized hamiltonian (penalty term vanishes for ground state)
        variational_diff = self.final_energy - self.full_solver.gse
        file.write(
            f"Deviation from exact energy: {variational_diff}\n"
            )

        file.write(f"Maxfev: {self.maxfev}\n")

        file.close()








    def draw_circuit(self):
        figure_subfolder = (f"./molecules/{self.molecule_name}"
                + f"/entanglement={self.ansatz.entanglement}"
                + f"/l={self.ansatz.num_layers}/")

        if not os.path.isdir(figure_subfolder):
            os.makedirs(figure_subfolder)
        try:
            self.circuit.decompose(reps=1).draw("mpl",
                filename=figure_subfolder+"circuit_ansatz.pdf",
                plot_barriers=True)

        except Exception as e:
            print(e)


    def _validate_inputs(self):
        assert isinstance(self.circuit, QuantumCircuit)
        assert isinstance(self.target_state, np.ndarray)


    def plot_result(self):
        def envelope(e_descent):
            lower_envelope = [e_descent[0]]

            for e in e_descent[1:]:
                if e < lower_envelope[-1]:
                    lower_envelope.append(e)
                else:
                    lower_envelope.append(lower_envelope[-1])

            return np.array(lower_envelope)

        infidelity_envelope = envelope(self.infidelity_descent)
        energy_envelope = envelope(self.energy_descent)

        plt.rcParams.update({
            "font.family": r"serif",  # use serif/main font for text elements
            "text.usetex": True,     # use inline math for ticks
            "pgf.rcfonts": False,    # don't setup fonts from rc parameters
            "pgf.preamble": "\n".join([
                 r"\usepackage{url}",            # load additional packages
                 r"\usepackage{lmodern}",   # unicode math setup
            ])
        })

        if self.molecule_name == "Fe3_NTA_doublet_CASSCF":
            MP2_diff = -12.1445816050894 - (-12.1868797976445)
        elif self.molecule_name == "Fe3_NTA_quartet_CASSCF":
            MP2_diff = -9.64899453554179 - (-9.72584117167342)

        print("MP2 diff:", MP2_diff)

        fig, ax = plt.subplots(figsize=(5,3))

        ax.set_yscale("log")

        ax.plot(infidelity_envelope, label="Infidelity", color="tab:blue")
        ax.plot(energy_envelope, label=r"$E(\theta) - E_0$ [hartree]", color="tab:orange")

        x_lim = ax.get_xlim()

        ax.plot([-8000, self.maxfev + 8000],
                [self.chemical_accuracy, self.chemical_accuracy],
                label="Chemical Accuracy",
                linestyle="dashed",
                color="black",
                )

        ax.plot([-8000, self.maxfev + 8000],
                [MP2_diff, MP2_diff],
                label="MP2 Accuracy",
                linestyle="dashed",
                color="gray",
                )

        ax.set_xlim(x_lim)

        ax.set_ylabel(r"Objective")
        ax.set_xlabel(r"Parameter updates")
        plt.legend()

        molecule_title = self.molecule_name.replace("_", " ")

        plt.title(f"{molecule_title}, Layers: {self.ansatz.num_layers}")

        plt.tight_layout()
        plt.savefig(self.subfolder_prefix + "objective_descent.pdf", backend="pgf")




# implementation of NFT optimizer
def nakanishi_fujii_todo(
    fun, x0, args=(), maxiter=None, maxfev=1024, reset_interval=32, eps=1e-32, callback=None, **_
):
    """
    Find the global minimum of a function using the nakanishi_fujii_todo
    algorithm [1].
    Args:
        fun (callable): ``f(x, *args)``
            Function to be optimized.  ``args`` can be passed as an optional item
            in the dict ``minimizer_kwargs``.
            This function must satisfy the three condition written in Ref. [1].
        x0 (ndarray): shape (n,)
            Initial guess. Array of real elements of size (n,),
            where 'n' is the number of independent variables.
        args (tuple, optional):
            Extra arguments passed to the objective function.
        maxiter (int):
            Maximum number of iterations to perform.
            Default: None.
        maxfev (int):
            Maximum number of function evaluations to perform.
            Default: 1024.
        reset_interval (int):
            The minimum estimates directly once in ``reset_interval`` times.
            Default: 32.
        eps (float): eps
        **_ : additional options
        callback (callable, optional):
            Called after each iteration.
    Returns:
        OptimizeResult:
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array. See
            `OptimizeResult` for a description of other attributes.
    Notes:
        In this optimization method, the optimization function have to satisfy
        three conditions written in [1].
    References:
        .. [1] K. M. Nakanishi, K. Fujii, and S. Todo. 2019.
        Sequential minimal optimization for quantum-classical hybrid algorithms.
        arXiv preprint arXiv:1903.12166.
    """

    x0 = np.asarray(x0)
    recycle_z0 = None
    niter = 0
    funcalls = 0

    while True:

        idx = niter % x0.size

        if reset_interval > 0:
            if niter % reset_interval == 0:
                recycle_z0 = None

        if recycle_z0 is None:
            z0 = fun(np.copy(x0), *args)
            funcalls += 1
        else:
            z0 = recycle_z0

        p = np.copy(x0)
        p[idx] = x0[idx] + np.pi / 2
        z1 = fun(p, *args)
        funcalls += 1

        p = np.copy(x0)
        p[idx] = x0[idx] - np.pi / 2
        z3 = fun(p, *args)
        funcalls += 1

        z2 = z1 + z3 - z0
        c = (z1 + z3) / 2
        a = np.sqrt((z0 - z2) ** 2 + (z1 - z3) ** 2) / 2
        b = np.arctan((z1 - z3) / ((z0 - z2) + eps * (z0 == z2))) + x0[idx]
        b += 0.5 * np.pi + 0.5 * np.pi * np.sign((z0 - z2) + eps * (z0 == z2))

        x0[idx] = b
        recycle_z0 = c - a

        niter += 1

        if callback is not None:
            callback(np.copy(x0))

        if maxfev is not None:
            if funcalls >= maxfev:
                break

        if maxiter is not None:
            if niter >= maxiter:
                break

    return OptimizeResult(
        fun=fun(np.copy(x0), *args), x=x0, nit=niter, nfev=funcalls, success=(niter > 1)
    )
