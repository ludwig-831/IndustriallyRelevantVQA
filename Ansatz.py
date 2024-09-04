from qiskit.circuit import QuantumCircuit
from copy import deepcopy
# variational circuit
from CustomExcitationPreserving import CustomExcitationPreserving

class Ansatz():

    def __init__(
        self,
        molecule_data: dict,
        num_layers: int,
        initial_state_str: str,
        entanglement: str = "full",
    ):

        self.molecule_data = molecule_data
        # get number of qubits
        self.num_qubits = molecule_data["num_qubits"]

        # number of spin up and spin down electrons
        self.num_s_up = molecule_data["num_s_up"]
        self.num_s_down = molecule_data["num_s_down"]


        self.num_layers = num_layers
        self.initial_state_str = initial_state_str
        self.entanglement = entanglement

        # validate inputs
        assert self._validate_inputs()

        # spin up qubits on even indices, spin down on odd ones
        self.initial_state = self.get_initial_state()

        self.circuit = self.set_circuit()

        self.num_parameters = self.circuit.num_parameters


    def get_initial_state(self):
        """
        Determines the initial states depending on the number of spin up
        and spin down electrons.
        """

        initial_state = QuantumCircuit(self.num_qubits)
        # spin up on even indices, spin down on odd ones
        for i in range(self.num_qubits):
            if int(self.initial_state_str[-(i+1)]) == 1:
                initial_state.x(i)

        initial_state = initial_state.decompose()
        return initial_state



    def set_circuit(self):
        """
        Creates the parameterized circuit.

        Returns:
            qiskit.QuantumCircuit
        """
        if self.num_layers == 0:
            return self.initial_state

        circ = deepcopy(self.initial_state)
        circ.barrier()

        # define qubit indices
        spin_up_qubits = [2*i for i in range(self.num_qubits//2)]
        spin_down_qubits = [2*i + 1 for i in range(self.num_qubits//2)]

        evolved_subsector_qubits = self.num_qubits//2

        # add layered ansatz
        for l in range(self.num_layers):
            excitation_preserving_spin_up = CustomExcitationPreserving(
                        num_qubits=evolved_subsector_qubits,
                        mode="Givens",
                        entanglement=self.entanglement,
                        reps=1,
                        initial_state=None,
                        insert_barriers=True,
                        skip_final_rotation_layer=True,
                        parameter_prefix=f"θ{l}",
                        )
            excitation_preserving_spin_down = CustomExcitationPreserving(
                        num_qubits=evolved_subsector_qubits,
                        mode="Givens",
                        entanglement=self.entanglement,
                        reps=1,
                        initial_state=None,
                        insert_barriers=True,
                        skip_final_rotation_layer=True,
                        parameter_prefix=f"φ{l}",
                        )



            circ.compose(excitation_preserving_spin_up,
                qubits=spin_up_qubits,
                inplace=True
                )
            circ.barrier()
            circ.compose(excitation_preserving_spin_down,
                qubits=spin_down_qubits,
                inplace=True
                )

            circ.barrier()

            # layer of CZs entangling the spin sectors
            for i in range(evolved_subsector_qubits):
                circ.cz(spin_up_qubits[i], spin_down_qubits[i])

            circ.barrier()

        return circ

    def _validate_inputs(self):

        assert isinstance(self.num_qubits, int) and self.num_qubits > 0
        # total number of qubits must be even
        assert self.num_qubits%2 == 0, "Total number of qubits must be even"

        # check valid number of electrons and qubits
        assert isinstance(self.num_s_up, int) and self.num_s_up >= 0
        assert isinstance(self.num_s_down, int) and self.num_s_down >= 0

        # must have at least one occupied spin orbital
        assert self.num_s_up + self.num_s_down > 0
        # can't have more electrons than available spin orbitals
        assert (self.num_s_up <= self.num_qubits//2) and (self.num_s_down <= self.num_qubits//2)

        # can't have more electrons than qubits
        assert self.num_s_up + self.num_s_down <= self.num_qubits, (
        f"Number of electrons (↑:{self.num_s_up}, ↓:{self.num_s_down})"
        +f" is larger than number of qubits ({self.num_qubits})."
        )

        # check number of layers
        assert isinstance(self.num_layers, int) and self.num_layers >= 0

        return True


    def print_attributes(self):
        attrs = vars(self)
        exclude = [
            "circuit",
            "initial_state",
            ]

        delim = 25*"="

        print("Attributes of Circuit:")
        print(delim)
        for key in attrs:
            if key not in exclude:
                print(key + ":", attrs[key])
        print(delim)
