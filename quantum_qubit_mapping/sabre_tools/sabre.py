from typing import Dict, List, Tuple
from networkx import Graph, DiGraph
from pyquil import Program
from pyquil.gates import Gate, SWAP
from pyquil.quilbase import Qubit  
import numpy as np


class SABRE:
    """
    SABRE mapper as described in
    “Tackling the Qubit Mapping Problem for NISQ‑Era Quantum Devices”
    (Li et al., 2019).  After the rewrites, every instruction stored in the
    output program is labelled with *physical* qubit indices.
    """

    # ---------------------------------------------------------------------
    #  Construction helpers
    # ---------------------------------------------------------------------

    def __init__(
        self,
        distance_matrix: np.ndarray,
        coupling_graph: Graph,
        heuristic: callable,
    ) -> None:
        self.distance_matrix = distance_matrix
        self.coupling_graph = coupling_graph
        self.heuristic = heuristic

    # ---------------------------------------------------------------------
    #  Internal utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _to_int(q) -> int:
        """Helper: Qubit | int  →  int"""
        return q.index if isinstance(q, Qubit) else int(q)

    def _gate_qubits(self, gate: Gate) -> Tuple[int, ...]:
        """
        Return the operands of *gate* as plain ints (logical or physical,
        depending on context).
        """
        return tuple(self._to_int(q) for q in gate.get_qubits())

    def logical_to_physical_gate(self, gate: Gate, mapping: Dict[int, int]) -> Gate:
        """
        Copy *gate* but replace every logical operand with its current
        physical location as per *mapping*.
        """
        phys_qubits = [Qubit(mapping[self._to_int(q)]) for q in gate.get_qubits()]
        return Gate(name=gate.name, params=gate.params, qubits=phys_qubits)

    def inst_swap_physical(self, swap_gate_logical: Gate, mapping: Dict[int, int]) -> Gate:
        """
        Convert a logical‑layer SWAP into a physical‑layer SWAP **and update the
        mapping in‑place**.
        """
        ql1, ql2 = self._gate_qubits(swap_gate_logical)
        qp1, qp2 = mapping[ql1], mapping[ql2]
        # update mapping
        mapping[ql1], mapping[ql2] = qp2, qp1
        # return physical instruction for the output program
        return SWAP(qp1, qp2)

    # ---------------------------------------------------------------------
    #  Main algorithm
    # ---------------------------------------------------------------------

    def execute_sabre_algorithm(
        self,
        front_layer_gates: List[tuple],
        qubit_mapping: Dict[int, int],
        circuit_dag: DiGraph,
    ) -> Tuple[Program, Dict[int, int]]:
        """
        Core scheduling / swap‑insertion loop.  Returns a PyQuil `Program`
        whose instructions are already expressed on physical qubits and the
        *final* logical→physical mapping.
        """
        decay_parameter = [0.001] * len(qubit_mapping)
        final_prog = Program()

        while front_layer_gates:
            # 1.  collect executable front‑layer gates under current mapping
            executable: List[tuple] = [
                gd for gd in front_layer_gates if self.is_gate_executable(gd[0], qubit_mapping)
            ]

            # 2A.  If some are executable, schedule them (physical labels) …
            if executable:
                for gd in executable:
                    front_layer_gates.remove(gd)

                    phys_gate = self.logical_to_physical_gate(gd[0], qubit_mapping)
                    final_prog.inst(phys_gate)

                    # push its successors whose all parents are now done
                    for succ in circuit_dag.successors(gd):
                        if not self.is_dependent_on_successors(succ, front_layer_gates):
                            front_layer_gates.append(succ)

            # 2B.  …otherwise pick ONE best SWAP and continue loop
            else:
                swap_candidates: List[Gate] = []
                for gd in front_layer_gates:
                    ctrl_l, tgt_l = self._gate_qubits(gd[0])
                    nbs_c, nbs_t = self.get_qubit_neighbours(ctrl_l, tgt_l, qubit_mapping)
                    swap_candidates += [SWAP(ctrl_l, n) for n in nbs_c]
                    swap_candidates += [SWAP(tgt_l, n) for n in nbs_t]

                scores = {
                    sw: self.heuristic(
                        front_layer_gates,
                        circuit_dag,
                        self.update_initial_mapping(sw, qubit_mapping),
                        self.distance_matrix,
                        sw,
                        decay_parameter,
                    )
                    for sw in swap_candidates
                }
                best_swap_logical = min(scores, key=scores.get)

                best_swap_physical = self.inst_swap_physical(best_swap_logical, qubit_mapping)
                final_prog.inst(best_swap_physical)

                decay_parameter = self.update_decay_parameter(best_swap_logical, decay_parameter)

                # re‑enter while loop with updated mapping
                continue

        return final_prog, qubit_mapping

    # ---------------------------------------------------------------------
    #  Executability helpers
    # ---------------------------------------------------------------------

    def is_gate_executable(self, gate: Gate, mapping: Dict[int, int]) -> bool:
        """
        True iff the (two‑qubit) *gate* acts on adjacent *physical* qubits.

        If the operands are present as KEYS in *mapping* we treat them as
        logical qubits; otherwise we assume they are already physical.
        """
        q1, q2 = self._gate_qubits(gate)

        if q1 in mapping and q2 in mapping:  # logical
            qp1, qp2 = mapping[q1], mapping[q2]
        else:  # physical
            qp1, qp2 = q1, q2

        return self.coupling_graph.has_edge(qp1, qp2)

    def is_dependent_on_successors(self, succ: tuple, front: List[tuple]) -> bool:
        succ_qubits = {self._to_int(q) for q in succ[0].get_qubits()}
        return any(
            succ_qubits & {self._to_int(x) for x in fg[0].get_qubits()}
            for fg in front
        )

    # ---------------------------------------------------------------------
    #  Neighbour utilities (mostly unchanged)
    # ---------------------------------------------------------------------

    def get_qubit_neighbours(
        self, ctrl_l: int, tgt_l: int, mapping: Dict[int, int]
    ) -> Tuple[List[int], List[int]]:
        ctrl_p, tgt_p = mapping[ctrl_l], mapping[tgt_l]
        nb_c, nb_t = (
            list(self.coupling_graph.neighbors(ctrl_p)),
            list(self.coupling_graph.neighbors(tgt_p)),
        )
        return (
            [self._logical_from_physical(p, mapping) for p in nb_c],
            [self._logical_from_physical(p, mapping) for p in nb_t],
        )

    @staticmethod
    def update_initial_mapping(sw: Gate, mapping: Dict[int, int]) -> Dict[int, int]:
        """Return a *copy* of mapping after hypothetically applying *sw*."""
        q1, q2 = sw.get_qubits()
        new_map = mapping.copy()
        new_map[q1], new_map[q2] = new_map[q2], new_map[q1]
        return new_map

    @staticmethod
    def update_decay_parameter(sw: Gate, decay: List[float]) -> List[float]:
        q1, q2 = sw.get_qubits()
        decay[q1] += 0.001
        decay[q2] += 0.001
        return decay

    # ---------------------------------------------------------------------
    #  Post‑run verification
    # ---------------------------------------------------------------------

    def rewiring_correctness(self, program: Program) -> Dict[Gate, Tuple[int, int]]:
        """
        Return a dict {bad_gate: (q1, q2)} for every two‑qubit gate that is not
        adjacent on the coupling graph.  The *program* is assumed to be
        physical (i.e. produced by this class).
        """
        bad = {}
        for ins in program.instructions:
            if ins.name == "SWAP" or len(ins.get_qubits()) < 2:
                continue
            q1, q2 = self._gate_qubits(ins)
            if not self.coupling_graph.has_edge(q1, q2):
                bad[ins] = (q1, q2)
        return bad

    # ---------------------------------------------------------------------
    #  Mini helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _logical_from_physical(p: int, mapping: Dict[int, int]) -> int:
        # reverse‑lookup helper (O(n) but n is small for NISQ)
        for log, phys in mapping.items():
            if phys == p:
                return log
        raise KeyError(p)
            
    def cnot_count(self, program: Program) -> int:
        """ Count the number of CNOT gates in the program.
        """
        count = 0
        for inst in program:
            if inst.name in ("CNOT", "CZ", "CPHASE"):
                count += 1
            elif inst.name == "SWAP":
                count += 3
        return count
