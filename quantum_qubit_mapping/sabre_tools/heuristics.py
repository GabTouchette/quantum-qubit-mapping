from networkx import DiGraph
from pyquil.gates import Gate
import numpy as np
import networkx as nx

def decay_weighted_distance_heuristic(F: list, circuit_dag: DiGraph, initial_mapping: dict, distance_matrix: np.matrix, swap_gate: Gate, decay_parameter: list) -> float:
    """
    The Decay-Weighted Distance (DWD) heuristic scores SWAP candidates by:
    1. Calculating average distance reduction for current executable gates (F)
    2. Incorporating a weighted lookahead via immediate successors (E)
    3. Applying decay-based weighting to discourage thrashing
    
    This is the baseline heuristic described in the original SABRE paper.
    """
    E = create_extended_successor_set(F, circuit_dag)
    min_score_swap_qubits = list(swap_gate.get_qubits())
    size_E = max(1, len(E))  # Prevent division by zero
    size_F = max(1, len(F))
    W = 0.5
    max_decay = max(decay_parameter[min_score_swap_qubits[0]], decay_parameter[min_score_swap_qubits[1]])
    f_distance = 0
    e_distance = 0
    for gate_details in F:
        f_distance += calculate_distance(gate_details, distance_matrix, initial_mapping)
    
    for gate_details in E:
        e_distance += calculate_distance(gate_details, distance_matrix, initial_mapping)

    f_distance = f_distance / size_F
    e_distance = W * (e_distance / size_E)
    H = max_decay * (f_distance + e_distance)
    return H

def calculate_distance(gate_details: tuple, distance_matrix: np.matrix, initial_mapping: dict) -> float:
    gate = gate_details[0]
    qubits = list(gate.get_qubits())
    return distance_matrix[initial_mapping.get(qubits[0]), initial_mapping.get(qubits[1])]

def create_extended_successor_set(F: list, circuit_dag: DiGraph) -> list: 
    E = list()
    for gate in  F:
        for gate_successor in circuit_dag.successors(gate):
            if len(E) <= 20:
                E.append(gate_successor)
    return E


def critical_path_heuristic(F: list, circuit_dag: DiGraph, initial_mapping: dict, 
                          distance_matrix: np.matrix, swap_gate: Gate, 
                          decay_parameter: list) -> float:
    """
    Prioritizes SWAPs that help gates on the critical path (longest path in DAG).
    Combines distance optimization with critical path awareness.
    """
    # Get original heuristic score as base
    base_score = decay_weighted_distance_heuristic(F, circuit_dag, initial_mapping, 
                                  distance_matrix, swap_gate, decay_parameter)
    
    # Calculate critical path impact
    critical_path_bonus = 0
    swap_qubits = list(swap_gate.get_qubits())
    
    # Check if swap helps any gate on critical path
    for gate in F:
        gate_qubits = list(gate[0].get_qubits())
        if set(swap_qubits) & set(gate_qubits):
            # Calculate how much this gate contributes to critical path
            path_length = len(list(nx.dag_longest_path(circuit_dag, gate)))
            critical_path_bonus += path_length
    
    # Normalize and weight the critical path factor
    cp_weight = 0.3  # Adjust based on experimentation
    critical_path_factor = 1 + (cp_weight * critical_path_bonus / (len(F) + 1))
    
    return base_score * critical_path_factor


def entanglement_aware_heuristic(F: list, circuit_dag: DiGraph, initial_mapping: dict,
                                distance_matrix: np.matrix, swap_gate: Gate,
                                decay_parameter: list) -> float:
    """
    Incorporates entanglement information by favoring SWAPs that:
    1. Maximize entanglement between qubits being moved
    2. Minimize disruption to existing entanglement
    """
    swap_qubits = list(swap_gate.get_qubits())
    
    # Calculate entanglement score for the swap
    ent_score = 0
    for gate in F:
        gate_qubits = list(gate[0].get_qubits())
        if set(swap_qubits) == set(gate_qubits):
            # Directly helps a CNOT - high value
            ent_score += 2
        elif len(set(swap_qubits) & set(gate_qubits)) == 1:
            # Affects one qubit of a CNOT - moderate value
            ent_score += 0.5
    
    # Get original heuristic score
    base_score = decay_weighted_distance_heuristic(F, circuit_dag, initial_mapping,
                                  distance_matrix, swap_gate, decay_parameter)
    
    # Combine with entanglement factor (weight can be tuned)
    ent_weight = 0.4
    return base_score * (1 - ent_weight) + (1 / (1 + ent_score)) * ent_weight


def lookahead_window_heuristic(F: list, circuit_dag: DiGraph, initial_mapping: dict,
                              distance_matrix: np.matrix, swap_gate: Gate,
                              decay_parameter: list, window_size: int = 5) -> float:
    """
    Extends the original heuristic with limited lookahead through a sliding window.
    Considers not just immediate gates but also upcoming gates in the circuit.
    """
    # Original heuristic components
    E = create_extended_successor_set(F, circuit_dag)
    swap_qubits = list(swap_gate.get_qubits())
    max_decay = max(decay_parameter[swap_qubits[0]], decay_parameter[swap_qubits[1]])
    
    # Calculate lookahead window
    lookahead_gates = []
    current_level = F.copy()
    
    for _ in range(window_size):
        next_level = []
        for gate in current_level:
            for successor in circuit_dag.successors(gate):
                next_level.append(successor)
        if not next_level:
            break
        lookahead_gates.extend(next_level)
        current_level = next_level
    
    # Calculate distances for different sets
    def calc_total_distance(gates, mapping):
        return sum(calculate_distance(g, distance_matrix, mapping) for g in gates)
    
    # Calculate with and without swap
    original_mapping = initial_mapping
    swapped_mapping = update_mapping_after_swap(swap_gate, initial_mapping.copy())
    
    # Current layer distances
    f_dist_original = calc_total_distance(F, original_mapping) / len(F)
    f_dist_swapped = calc_total_distance(F, swapped_mapping) / len(F)
    
    # Extended layer distances
    e_dist_original = 0.5 * calc_total_distance(E, original_mapping) / (len(E) or 1)
    e_dist_swapped = 0.5 * calc_total_distance(E, swapped_mapping) / (len(E) or 1)
    
    # Lookahead distances
    l_dist_original = 0.2 * calc_total_distance(lookahead_gates, original_mapping) / (len(lookahead_gates) or 1)
    l_dist_swapped = 0.2 * calc_total_distance(lookahead_gates, swapped_mapping) / (len(lookahead_gates) or 1)
    
    # Combined score
    original_score = f_dist_original + e_dist_original + l_dist_original
    swapped_score = f_dist_swapped + e_dist_swapped + l_dist_swapped
    
    return max_decay * (original_score - swapped_score)

def update_mapping_after_swap(swap_gate: Gate, mapping: dict) -> dict:
    """Helper function to simulate mapping after swap"""
    q1, q2 = swap_gate.get_qubits()
    mapping[q1], mapping[q2] = mapping[q2], mapping[q1]
    return mapping