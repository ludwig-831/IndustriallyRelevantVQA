import numpy as np
from ConfigurationRecovery import ConfigurationRecovery
from OperatorLoader import OperatorLoader
import yaml
from copy import deepcopy
from tqdm import tqdm

delim = 15*"="

chemical_accuracy = 4/2625.49963948

molecule_name = "Fe3_NTA_doublet_CASSCF" # "Fe3_NTA_quartet_CASSCF"
print(molecule_name)
final_run = "Run 62" # Run 62 was the final sample of the optimized state
path_to_molecule = f"molecules/{molecule_name}/"

if molecule_name == "Fe3_NTA_doublet_CASSCF":
    run = "240620_02"
else:
    run = "240620_01"



path_to_data = "Results/" + path_to_molecule + f"entanglement=full/l=1/{run}/"

operator_loader = OperatorLoader(molecule_name)
molecule_data = operator_loader.molecule_data

GSE = molecule_data["gse"]

# load entire hardware results
with open(path_to_data + "full_result.yml", "r") as f:
    samples = yaml.safe_load(f)

final_samples = samples[final_run] # pick the final run


gse_list = []
energy_list = []

iterations = 10
for _ in tqdm(range(iterations)):

    reco = ConfigurationRecovery(final_samples, K=10, operator_loader=operator_loader)

    reco.run_setup()

    original_strings = deepcopy(reco.correct_bitstrings)

    pre_gse = (reco.get_groundstate_energy(reco.correct_bitstrings))

    reco.run()

    post_gse = (reco.get_groundstate_energy(reco.correct_bitstrings))

    energy_list.append(post_gse)
    gse_list.append(post_gse - GSE)


print(np.min(energy_list))
print(np.min(gse_list))
