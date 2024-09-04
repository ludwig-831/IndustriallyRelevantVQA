# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ExcitationPreserving 2-local circuit."""

from typing import Union, Optional, List, Tuple, Callable, Any
from numpy import pi

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library.standard_gates import RZGate
from qiskit.circuit.library import TwoLocal


class CustomExcitationPreserving(TwoLocal):
    r"""The heuristic excitation-preserving wave function ansatz.

    The ``ExcitationPreserving`` circuit preserves the ratio of :math:`|00\rangle`,
    :math:`|01\rangle + |10\rangle` and :math:`|11\rangle` states. To this end, this circuit
    uses two-qubit interactions of the form

    .. math::

        \newcommand{\th}{\theta/2}

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & \cos\left(\th\right) & \sin\left(\th\right) & 0 \\
        0 & -\sin\left(\th\right) & \cos\left(\th\right) & 0 \\
        0 & 0 & 0 & 1
        \end{pmatrix}

    for the mode ``'Givens'``.

    Note that other wave functions, such as UCC-ansatzes, are also excitation preserving.
    However these can become complex quickly, while this heuristically motivated circuit follows
    a simpler pattern.

    This trial wave function consists of layers of :math:`Z` rotations with 2-qubit entanglements.
    The entangling is creating using :math:`XX+YY` rotations and optionally a controlled-phase
    gate for the mode ``'fsim'``.

    See :class:`~qiskit.circuit.library.RealAmplitudes` for more detail on the possible arguments
    and options such as skipping unentanglement qubits, which apply here too.

    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        mode: str = "Givens",
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "full",
        reps: int = 3,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
        parameter_prefix: str = "θ",
        insert_barriers: bool = False,
        initial_state: Optional[Any] = None,
        name: str = "ExcitationPreserving",
    ) -> None:
        """Create a new ExcitationPreserving 2-local circuit.

        Args:
            num_qubits: The number of qubits of the ExcitationPreserving circuit.
            mode: Choose the entangler mode, can be `'iswap'` or `'fsim'`.
            reps: Specifies how often the structure of a rotation layer followed by an entanglement
                layer is repeated.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                See the Examples section of :class:`~qiskit.circuit.library.TwoLocal` for more
                detail.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If True, a rotation layer is added at the end of the
                ansatz. If False, no rotation layer is added. Defaults to True.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use :class:`~qiskit.circuit.ParameterVector`.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.

        Raises:
            ValueError: If the selected mode is not supported.
        """
        supported_modes = ["Givens"]

        assert mode in supported_modes

        if mode == "Givens":
            # https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3/pdf
            # Fig 2.
            theta = Parameter("θ")
            swap = QuantumCircuit(2, name="Interaction")

            swap.h(0)
            swap.cx(0, 1)

            swap.ry(theta/2, 0)
            swap.ry(theta/2, 1)

            swap.cx(0, 1)
            swap.h(0)

            swap = swap.to_gate(label=r"G")

        super().__init__(
            num_qubits=num_qubits,
            rotation_blocks=None,
            entanglement_blocks=swap,
            entanglement=entanglement,
            reps=reps,
            skip_unentangled_qubits=skip_unentangled_qubits,
            skip_final_rotation_layer=skip_final_rotation_layer,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            initial_state=initial_state,
            name=name,
        )

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Return the parameter bounds.

        Returns:
            The parameter bounds.
        """
        return self.num_parameters * [(-pi, pi)]
