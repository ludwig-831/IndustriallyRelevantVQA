import numpy as np
import json
from qiskit.circuit import QuantumCircuit

#####################################################################################################################
########################## Useful helper objects ####################################################################
#####################################################################################################################
def hit_by(O,P):
    """ Returns whether o is hit by p """
    for o,p in zip(O,P):
        if not (o==0 or p==0 or o==p):
            return False
    return True

# useful conversion between character and int to denote single qubit Pauli operators
char_to_int = {"I":0,"X":1,"Y":2,"Z":3}
int_to_char = {item: key for key,item in char_to_int.items()}

#####################################################################################################################
########################## Measurement scheme and helper classes ####################################################
#####################################################################################################################
########################## Details can be found at https://gitlab.com/GreschAI/shadowgrouping #######################
########################## and at https://arxiv.org/abs/2301.03385 ##################################################
#####################################################################################################################
#####################################################################################################################

class Measurement_scheme:
    """ Parent class for measurement schemes. Requires
        sparse_pauli: instance of SparsePauliOperator from which the decomposition is inferred
        epsilon:         Absolute error threshold, see child methods for an individual interpretation
    """

    def __init__(self,sparse_pauli):
        self.__get_values_from_sparse_pauli(sparse_pauli)
        M,n = self.obs.shape
        self.num_obs       = M
        self.num_qubits    = n
        self.N_hits        = np.zeros(M,dtype=int)
        return

    def find_setting(self):
        pass

    def __get_values_from_sparse_pauli(self,sparse_pauli):
        obs, weights = zip(*sparse_pauli.to_list())
        obs, weights = list(obs), list(weights)

        # filter out the offset energy
        idx = -1
        offset = 0
        for i,(o,w) in enumerate(zip(obs,weights)):
            if o == "I"*len(o):
                offset = w.real
                idx = i
        if idx >= 0:
            obs.pop(idx)
            weights.pop(idx)
        self.w = np.asarray(weights).real
        # go from Pauli label to integer representation
        self.obs = np.asarray([[char_to_int[c] for c in o] for o in obs],dtype=int)
        self.offset = offset
        return

    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        return


#####################################################################################################################
########################## Energy estimator class ###################################################################
#####################################################################################################################

class Energy_estimator():
    """ Convenience class that holds both a measurement scheme and a VQA instance.
        The main workflow consists of proposing the next (few) measurement settings and measuring them in the VQA.
        Furthermore, it tracks all measurement settings and their respective outcomes (of value +/-1 per qubit).
        Based on these values, the current energy estimate can be calculated.

        Inputs:
        - measurement_scheme, see class Measurement_Scheme and subclasses for information.
        - state, see class StateSampler.
        - Energy offset (defaults to 0) for the energy estimation.
          This typically consists of the identity term in the corresponding Hamiltonian decomposition.
        - repeats (int), counting how often the same measure()-step has to be repeated. Can be used to gather statistics for the same measurement proposal
        - spin_core (float or None), if known, post-selection is applied to correct towards the right spin value. This increases the number of unique circuits two-fold
    """
    def __init__(self,measurement_scheme):
        self.measurement_scheme = measurement_scheme
        self.offset       = measurement_scheme.offset
        # convenience counters to keep track of measurements settings and respective outcomes
        self.settings_dict = {}
        self.settings_buffer = {}
        self.num_settings = 0
        self.num_outcomes = 0
        self.measurement_scheme.reset()
        self.lookup_circuits = {"X": ["i","h"], "Y": ["sdg","h"]}
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N    = np.zeros(len(self.running_avgs),dtype=int)
        return

    @property
    def H_decomp(self):
        return (self.measurement_scheme.obs,self.measurement_scheme.w,self.offset)

    def reset(self):
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N    = np.zeros(len(self.running_avgs),dtype=int)
        self.settings_dict = {}
        self.settings_buffer = {}
        self.num_settings, self.num_outcomes = 0, 0
        self.measurement_scheme.reset()
        return

    def clear_outcomes(self):
        self.settings_buffer = self.settings_dict.copy()
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N    = np.zeros(len(self.running_avgs),dtype=int)
        self.num_outcomes = 0
        return

    def __setting_to_str(self,p):
        out = ""
        for c in p:
            out += int_to_char[c]
        return out

    def __settings_to_dict(self,settings):
        # run all the last prepared measurement settings
        # from the <settings>-list, fetch the unique settings and their respective counts and put them into settings_dict
        dicts = (self.settings_dict,self.settings_buffer)
        unique_settings, counts = np.unique(settings,axis=0,return_counts=True)
        for setting,nshots in zip(unique_settings,counts):
            setting = self.__setting_to_str(setting)
            for diction in dicts:
                val = diction.get(setting,None)
                if val is not None:
                    val["nshots"] += nshots
                else:
                    diction[setting] = {"nshots": nshots, "circuit": self.circuit_from_pauli(setting)}
        return

    def circuit_from_pauli(self,pauli):
        num_qubits = len(pauli)
        circuit = QuantumCircuit(num_qubits)

        for i,c in enumerate(reversed(pauli)):
            for instruction in self.lookup_circuits.get(c,["i","i"]):
                getattr(circuit,instruction)(i)
        circuit.measure_all()
        return circuit

    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        assert isinstance(num_steps,int) or isinstance(num_steps,np.int64), "<num_steps> has to be integer but was {}".format(type(num_steps))
        assert num_steps > 0, "<num_steps> has to be positive but was {}".format(num_steps)
        settings = []
        for i in range(num_steps):
            p = self.measurement_scheme.find_setting()
            settings.append(p)
        settings = np.array(settings)
        self.num_settings += num_steps
        self.__settings_to_dict(settings)
        return

    def __qiskit_dict_to_outcomes(self,qiskit_counts,N):
        out = np.zeros((N,self.measurement_scheme.num_qubits),dtype=int)
        ind = 0
        for bitstring, reps in qiskit_counts.items():
            # map 0/1 to 1/-1, respectively
            arr = -2*np.array([int(b) for b in bitstring]).reshape((1,-1)) + 1
            out[ind:ind+reps] = np.repeat(arr,repeats=reps,axis=0)
            ind += reps
        assert ind==N, "Something wrong in __qiskit_dict_to_outcome"
        return out

    def get_outcomes(self,outcomes):
        """ Add the corresponding outcomes to the settings in <settings_buffer>. """
        assert isinstance(outcomes,dict), "Outcomes has to be provided in form of a dictionary but was {}".format(type(outcomes))
        num_meas = self.num_settings - self.num_outcomes
        if num_meas == 0:
            print("Trying to feed more outcomes than allocated measurement settings. Please allocate measurements first by calling propose_next_settings() first.")
            print("Ignoring outcomes.")
            return
        for key,val in outcomes.items():
            # check whether outcome correspond to any assigned key and that the outcomes have the right array shape
            temp = self.settings_buffer.get(key,None)
            if temp is None:
                print("No outcomes required for setting {}! Skipping these values.".format(key))
                continue
            val = self.__qiskit_dict_to_outcomes(val,temp["nshots"])
            temp = (temp["nshots"],self.measurement_scheme.num_qubits)
            assert np.allclose(val.shape,temp), "Wrong shape of outcomes for setting {}. Should have been {}.".format(key,temp)
        for setting,temp in self.settings_buffer.items():
            # check whether a key exists in outcomes for every required setting. If it exists, it is already checked for the right shape above
            outcome = outcomes.get(setting,None)
            assert outcome is not None, "No outcomes recorded for setting {}.".format(setting)
            outcome = self.__qiskit_dict_to_outcomes(outcome,temp["nshots"])
            N,n = outcome.shape
            # write into running_avgs
            for i,o in enumerate(self.measurement_scheme.obs):
                if not hit_by(o,[char_to_int[c] for c in setting]):
                    continue
                mask = np.zeros((N,n),dtype=int)
                mask += (o == 0)[np.newaxis,:]
                temp = outcome.copy()
                temp[mask.astype(bool)] = 1 # set to 1 if outside the support of the respective hit observable to mask it out
                self.running_avgs[i] = ( self.running_avgs[i]*self.running_N[i] + np.prod(temp,axis=1).sum() ) / (self.running_N[i] + N)
                self.running_N[i] += N
        self.num_outcomes = self.num_settings
        self.settings_buffer = {}
        return

    def get_energy(self):
        """ Takes the current outcomes and estimates the corresponding energy. """
        energy = np.sum(self.measurement_scheme.w*self.running_avgs)
        return energy + self.offset

    def save_settings(self,filename):
        data = {key: int(val["nshots"]) for key,val in self.settings_dict.items()}
        data["Ntotal"] = self.num_settings
        with open(filename, 'w') as f:
            json.dump(data, f)
        return

    def load_settings(self,filename):
        self.reset()
        with open(filename, 'r') as f:
            data = json.load(f)
        self.num_settings = data.pop("Ntotal")
        self.settings_dict = {key: {"nshots": val, "circuit": self.circuit_from_pauli(key)} for key,val in data.items()}
        self.settings_buffer = self.settings_dict.copy()
        return


class Zstring_Estimator(Energy_estimator):

    def __init__(self,measurement_scheme):
        super().__init__(measurement_scheme)
        no_commuting_obs = True
        self.setting     = [3]*self.measurement_scheme.num_qubits
        self.setting_str = "Z"*self.measurement_scheme.num_qubits
        for o in self.measurement_scheme.obs:
            if hit_by(o,self.setting):
                no_commuting_obs = False
        assert not no_commuting_obs, "None of the observables provided can be measured in the computational basis. Aborted."
        self.E_samples = []

    def reset(self):
        super().reset()
        self.E_samples = []

    def propose_next_settings(self,num_settings=1):
        self.measurement_scheme.N_hits[[hit_by(o,self.setting) for o in self.measurement_scheme.obs]] += num_settings
        for diction in (self.settings_dict,self.settings_buffer):
            val = diction.get(self.setting_str,None)
            if val is not None:
                val["nshots"] += num_settings
            else:
                diction[self.setting_str] = {"nshots": num_settings, "circuit": self.circuit_from_pauli(self.setting_str)}
        self.num_settings += num_settings
        return

    def get_outcomes(self,outcomes):
        assert isinstance(outcomes,dict), "Outcomes has to be provided in form of a dictionary but was {}".format(type(outcomes))
        num_meas = self.num_settings - self.num_outcomes
        if num_meas == 0:
            print("Trying to feed more outcomes than allocated measurement settings. Please allocate measurements first by calling propose_next_settings() first.")
            print("Ignoring outcomes.")
            return
        if len(outcomes) > 1:
            print("Warning! This wrapper class only takes into account outcomes for computational basis measurements.")
            print("All other outcomes will be neglected")
        outcome = outcomes.get(self.setting_str,None)
        if outcome is None:
            print("No relevant data found in outcomes. Skipping.")
            return
        # check whether outcome correspond to any assigned key and that the outcomes have the right array shape
        temp = self.settings_buffer[self.setting_str]
        outcome = self._Energy_estimator__qiskit_dict_to_outcomes(outcome,temp["nshots"])
        temp = (temp["nshots"],self.measurement_scheme.num_qubits)
        assert np.allclose(outcome.shape,temp), "Wrong shape of outcomes for Z-string setting: Should have been {} but was {}.".format(temp,outcome.shape)

        N,n = outcome.shape
        E_vals = np.zeros(N)
        # write into running_avgs and track individual values
        # the loop in init ensures that at least one value is provided for E_vals
        for i,(o,w) in enumerate(zip(self.measurement_scheme.obs,self.measurement_scheme.w)):
            if not hit_by(o,self.setting):
                continue
            mask = np.zeros((N,n),dtype=int)
            mask += (o == 0)[np.newaxis,:]
            temp = outcome.copy()
            temp[mask.astype(bool)] = 1 # set to 1 if outside the support of the respective hit observable to mask it out
            E_val = np.prod(temp,axis=1) # of size (N,) afterwards
            E_vals += E_val*w
            self.running_avgs[i] = ( self.running_avgs[i]*self.running_N[i] + E_val.sum() ) / (self.running_N[i] + N)
            self.running_N[i] += N
        self.E_samples = np.append(self.E_samples,E_vals)
        self.num_outcomes = self.num_settings
        self.settings_buffer = {}
        return

    def get_uncertainty(self):
        if self.num_outcomes == 0:
            print("Warning! No outcomes recorded to yield a meaningful uncertainty on estimate. Returned -1")
            return -1
        return np.std(self.E_samples)/np.sqrt(self.num_outcomes)
