from networkx import DiGraph
from pyquil.gates import Gate
import numpy as np
import networkx as nx
from collections import Counter

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
    size_E = max(1, len(E))
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


def decay_weighted_distance_heuristic_upgraded(F: list, circuit_dag: DiGraph, initial_mapping: dict, distance_matrix: np.matrix, swap_gate: Gate, decay_parameter: list) -> float:
    """
    Problem: once the distance to all front gates is 0, DWD no longer distinguishes 
    good from bad swaps – it may happily shuffle qubits that are already neighbours 
    (thrashing).
    """
    E = create_extended_successor_set(F, circuit_dag)
    min_score_swap_qubits = list(swap_gate.get_qubits())
    size_E = max(1, len(E))
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
    
    # Penalise breaking satisfied gates
    if f_distance == 0:
        temp_map = update_mapping_after_swap(swap_gate, initial_mapping.copy())
        post_dist = sum(calculate_distance(g, distance_matrix, temp_map)
                        for g in F) / size_F
        if post_dist > 0:
            H += 30 # arbitrary penalty, tested with 5, 10, 50 and 30 was best
    return H


def critical_path_heuristic(F: list, circuit_dag: DiGraph, initial_mapping: dict, 
                          distance_matrix: np.matrix, swap_gate: Gate, 
                          decay_parameter: list) -> float:
    """
    Prioritizes SWAPs that help gates on the critical path (longest path in DAG).
    Combines distance optimization with critical path awareness.
    """
    base_score = decay_weighted_distance_heuristic(F, circuit_dag, initial_mapping, 
                                  distance_matrix, swap_gate, decay_parameter)
    
    critical_path_bonus = 0
    swap_qubits = list(swap_gate.get_qubits())
    
    for gate in F:
        gate_qubits = list(gate[0].get_qubits())
        if set(swap_qubits) & set(gate_qubits):
            path_length = len(list(nx.dag_longest_path(circuit_dag, gate)))
            critical_path_bonus += path_length
    
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
    
    ent_score = 0
    for gate in F:
        gate_qubits = list(gate[0].get_qubits())
        if set(swap_qubits) == set(gate_qubits):
            ent_score += 2
        elif len(set(swap_qubits) & set(gate_qubits)) == 1:
            ent_score += 0.5
    
    base_score = decay_weighted_distance_heuristic(F, circuit_dag, initial_mapping,
                                  distance_matrix, swap_gate, decay_parameter)
    
    max_possible = 2 * len(F)
    rel = ent_score / (max_possible or 1)
    ent_factor = 1 + 0.5 * np.tanh(3*(rel-0.3))
    return base_score * ent_factor


def lookahead_window_heuristic(F: list, circuit_dag: DiGraph, initial_mapping: dict,
                              distance_matrix: np.matrix, swap_gate: Gate,
                              decay_parameter: list) -> float:
    """
    Extends the original heuristic with limited lookahead through a sliding window.
    Considers not just immediate gates but also upcoming gates in the circuit.
    """
    E = create_extended_successor_set(F, circuit_dag)
    swap_qubits = list(swap_gate.get_qubits())
    max_decay = max(decay_parameter[swap_qubits[0]], decay_parameter[swap_qubits[1]])
    
    lookahead_gates = []
    current_level = F.copy()
    window_size = max(3, len(F))
    
    for _ in range(window_size):
        next_level = []
        for gate in current_level:
            for successor in circuit_dag.successors(gate):
                next_level.append(successor)
        if not next_level:
            break
        lookahead_gates.extend(next_level)
        current_level = next_level
    
    def calc_total_distance(gates, mapping):
        return sum(calculate_distance(g, distance_matrix, mapping) for g in gates)
    
    original_mapping = initial_mapping
    swapped_mapping = update_mapping_after_swap(swap_gate, initial_mapping.copy())
    
    f_dist_original = calc_total_distance(F, original_mapping) / len(F)
    f_dist_swapped = calc_total_distance(F, swapped_mapping) / len(F)    
    e_dist_swapped = 0.5 * calc_total_distance(E, swapped_mapping) / (len(E) or 1)
    l_dist_swapped = 0.2 * calc_total_distance(lookahead_gates, swapped_mapping) / (len(lookahead_gates) or 1)

    swapped_total = f_dist_swapped + e_dist_swapped + l_dist_swapped
    
    return max_decay * swapped_total + 1e-6 * (f_dist_original - f_dist_swapped)

def update_mapping_after_swap(swap_gate: Gate, mapping: dict) -> dict:
    """Helper function to simulate mapping after swap"""
    q1, q2 = swap_gate.get_qubits()
    mapping[q1], mapping[q2] = mapping[q2], mapping[q1]
    return mapping


def order_cost(mapping: dict[int, int]) -> int:
    return sum(abs(l - p) for l, p in mapping.items())

def extended_successor_set(F, dag, limit=20):
    S = []
    for n in F:
        for succ in dag.successors(n):
            if len(S) < limit:
                S.append(succ)
    return S

def hybrid_lexi_heuristic(F, circuit_dag, mapping, distance_matrix,
                          swap_gate, decay_parameter):
    """Tuple score; SABRE will pick min() lexicographically."""
    new_map = update_mapping_after_swap(swap_gate, mapping.copy())

    def layer_cost(nodes, mp):
        if not nodes:
            return 0
        tot = 0
        for nd in nodes:
            g = nd[0]
            q1, q2 = [q.index for q in g.qubits]
            tot += distance_matrix[mp[q1], mp[q2]]
        return tot / len(nodes)

    cost_F = layer_cost(F, mapping)
    cost_E = layer_cost(extended_successor_set(F, circuit_dag), mapping)
    order_improv = order_cost(mapping) - order_cost(new_map)

    return (cost_F, cost_E, -order_improv)