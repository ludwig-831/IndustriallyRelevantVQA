import numpy as np
import json
import sys
sys.path.append("../")
from qiskit_aer.noise import (NoiseModel, depolarizing_error, ReadoutError)

from SolverAtParticleNumber import SolverAtParticleNumber
from OperatorLoader import OperatorLoader
from NoisyVQE import NoisyVQE
from Ansatz import Ansatz
delim = 15*"="

# create noise model for noisy simulations
noise_model = NoiseModel()
error_1q = depolarizing_error(0.005, 1)
error_2q = depolarizing_error(0.025, 2)

noise_model.add_all_qubit_quantum_error(error_1q, ["rx", "ry", "r"])
noise_model.add_all_qubit_quantum_error(error_2q, ["rxx"])
basis_gates = ["rxx", "rx", "ry", "rz", "r", "id"]

# backend options to be passed to NoisyVQE
backend_options = {}
backend_options["noise_model"] = noise_model
backend_options["coupling_map"] = None # no coupling map -> all-to-all connectivity
backend_options["basis_gates"] = basis_gates


# Choose molecule
molecule_name = "Fe3_NTA_doublet_CASSCF"
print(molecule_name)

# load molecule data
path_to_data = f"./molecules/{molecule_name}/"
with open(path_to_data + "data_file.json", "r") as read_file:
    molecule_data = json.load(read_file)

# Measurement settings
Nshots = int(6e2) # shots per parameter update
Nmax = 1 # number of unique circuit settings
# ShadowGrouping settings
delta  = 0.1 # or None for deactivation of truncation, sets confidence
eps_SG = 0.1 # <-- the float does not really matter here

Nshots_final_state = int(1e4) # number of shots for sampling the circuit
# with optimized parameters in the end

measurement_config = {
    "Nshots": Nshots,
    "Nmax": Nmax,
    "eps_SG": eps_SG,
    "Nshots_final_state": Nshots_final_state,
}


# Circuit ansatz settings
circuit_ansatz = "ExcitationPreserving"
mode = "GivensCZ"
entanglement = "full"
num_layers = 1
parity_preserving = False

# Number of parameter updates, optimizations ends after maxfev updates
maxfev = int(1e2)

# OperatorLoader loads Hamiltonian etc.
operator_loader = OperatorLoader(molecule_name)
# solver gets ground state (energy) etc.
solver = SolverAtParticleNumber(
    operator_loader,
)
solver.solve()
# Hartree-Fock string for Circuit initialization
hf_str = solver.get_hartree_string()

# Ansatz contains circuit
ansatz = Ansatz(
    molecule_data=molecule_data,
    num_layers=num_layers,
    initial_state_str=hf_str,
)

ansatz.print_attributes()


# NoisyVQE class collects everything for optimization
# also handles plotting and saving results
vqe = NoisyVQE(
    measurement_config=measurement_config,
    backend_options=backend_options,
    ansatz=ansatz,
    molecule_name=molecule_name,
    full_solver=solver,
    operator_loader=operator_loader,
    )


# initial set of variational parameters
x0 = [0 for _ in range(ansatz.circuit.num_parameters)]

optimizer_result = vqe.optimize(
                maxfev=maxfev,
                x0=x0
                )
