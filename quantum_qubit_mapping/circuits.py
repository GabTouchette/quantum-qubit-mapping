from pyquil import Program
from pyquil.gates import CNOT, H, SWAP, CZ, CPHASE
import networkx as nx
import numpy as np

def create_4mod5_v1_22():
    """4mod5-v1_22 (4 qubits)"""
    circuit = Program()
    circuit.inst(CNOT(0, 1))
    circuit.inst(CNOT(2, 3))
    circuit.inst(CNOT(1, 3))
    circuit.inst(CNOT(1, 2))
    circuit.inst(CNOT(2, 3))
    circuit.inst(CNOT(0, 3))
    for _ in range(3):
        circuit.inst(CNOT(0, 1))
        circuit.inst(CNOT(2, 3))
        circuit.inst(CNOT(1, 3))
    
    coupling_graph = nx.path_graph(4)
    return circuit, coupling_graph

def create_mod5mils_65():
    """mod5mils_65 (4 qubits)""" 
    circuit = Program()
    for i in range(3):
        circuit.inst(CNOT(i, i+1))
    circuit.inst(CNOT(0, 2))
    circuit.inst(CNOT(1, 3))
    for i in range(3, 0, -1):
        circuit.inst(CNOT(i, i-1))
    circuit.inst(CNOT(0, 2))
    circuit.inst(CNOT(1, 3))
    
    coupling_graph = nx.path_graph(4)
    return circuit, coupling_graph

def create_decod24_v2_43():
    """decod24-v2_43 (4 qubits)"""
    circuit = Program()
    for _ in range(9):
        circuit.inst(CNOT(0, 1))
        circuit.inst(CNOT(0, 2))
        circuit.inst(CNOT(0, 3))
        circuit.inst(CNOT(1, 2))
        circuit.inst(CNOT(1, 3))
        circuit.inst(CNOT(2, 3))
    
    coupling_graph = nx.path_graph(4)
    return circuit, coupling_graph

def create_4gt13_92():
    """4gt13_92 (5 qubits)"""
    circuit = Program()
    for _ in range(7):
        circuit.inst(CNOT(0, 1))
        circuit.inst(CNOT(2, 3))
        circuit.inst(CNOT(1, 2))
        circuit.inst(CNOT(3, 4))
        circuit.inst(CNOT(0, 4))
        circuit.inst(CNOT(1, 3))
    
    coupling_graph = nx.path_graph(5)
    return circuit, coupling_graph

def create_ising_model_10():
    """Create valid Ising model with explicit dependencies"""
    n_qubits = 10
    circuit = Program()
    coupling_graph = nx.path_graph(n_qubits)
    
    for _ in range(3):
        # Forward layer
        for q in range(n_qubits - 1):
            circuit.inst(CZ(q, q+1))
        for q in range(n_qubits-1, 0, -1):
            circuit.inst(CZ(q-1, q))
    
    return circuit, coupling_graph

def create_ising_model_13():
    """Create valid Ising model with explicit dependencies"""
    n_qubits = 13
    circuit = Program()
    coupling_graph = nx.path_graph(n_qubits)
    
    for _ in range(3):
        # Forward layer
        for q in range(n_qubits - 1):
            circuit.inst(CZ(q, q+1))
        for q in range(n_qubits-1, 0, -1):
            circuit.inst(CZ(q-1, q))
    
    return circuit, coupling_graph

def create_ising_model_16():
    """Create valid Ising model with explicit dependencies"""
    n_qubits = 16
    circuit = Program()
    coupling_graph = nx.path_graph(n_qubits)
    
    for _ in range(3):
        # Forward layer
        for q in range(n_qubits - 1):
            circuit.inst(CZ(q, q+1))
        for q in range(n_qubits-1, 0, -1):
            circuit.inst(CZ(q-1, q))
    
    return circuit, coupling_graph


def create_qft_10():
    """Quantum Fourier Transform on 10 qubits."""
    n_qubits = 10
    prog = Program()

    for k in range(n_qubits):
        # prog.inst(H(k)) # Only include 2-qubit gates
        for j in range(1, n_qubits - k):
            angle = np.pi / (2 ** j)
            prog.inst(CPHASE(angle, k + j, k))

    for i in range(n_qubits // 2):
        prog.inst(SWAP(i, n_qubits - i - 1))

    coupling_graph = nx.path_graph(n_qubits)
    return prog, coupling_graph
