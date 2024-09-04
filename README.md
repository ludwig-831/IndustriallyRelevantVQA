# How to use
To execute the parameter optimization, run noisy_vqe.py
Change the backend from qiskit's Aer simulator backend to the hardware backend you are using.
You will find the results in a subfolder of "molecules/", the path depends on the settings you chose.
The molecule names "Fe3_NTA_doublet_CASSCF" and "Fe3_NTA_quartet_CASSCF" denote the low and intermediate
spin states.
Change the molecule name to run the process for the different cases.

Make sure that the raw data is saved by your backend as well.
The NoisyVQE class does not do this, as the backend we used did not require it.
The raw data is needed to execute run_configurationRecovery.py and needs to be in the same format as the data
in Results/.../full_result.yml.
Alternatively one can change the way data is loaded.
