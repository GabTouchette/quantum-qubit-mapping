"""
Light‑weight state / circuit analysis utilities.
Keeps everything in PyQuil, no full‑state simulation required.
"""

from collections import Counter
import numpy as np
import networkx as nx
from pyquil import Program


# ---------------------------------------------------------------------
# 1. Interaction graph  (weighted logical‑qubit graph)
# ---------------------------------------------------------------------
def build_interaction_graph(prog: Program) -> nx.Graph:
    g = nx.Graph()
    for inst in prog.instructions:
        if len(inst.qubits) == 2:
            q1, q2 = [q.index for q in inst.qubits]
            if g.has_edge(q1, q2):
                g[q1][q2]["w"] += 1
            else:
                g.add_edge(q1, q2, w=1)
    return g


# ---------------------------------------------------------------------
# 2. Distance‑aware “best effort” initial layout
#    Path‑graph hardware gets an O(N log N) sort; otherwise Hungarian.
# ---------------------------------------------------------------------
def compute_initial_layout(inter_graph: nx.Graph, coupling_graph: nx.Graph) -> dict:
    n = len(coupling_graph)
    if nx.is_tree(coupling_graph) and coupling_graph.size() == n - 1:
        # ------ hardware is a line / chain ------
        # Sort logical qubits by *weighted degree* (heavy movers near centre)
        weighted_deg = Counter({v: 0 for v in inter_graph.nodes})
        for u, v, d in inter_graph.edges(data="w"):
            weighted_deg[u] += d
            weighted_deg[v] += d
        chain_order = [q for q, _ in weighted_deg.most_common()]
        # if fewer logical than physical qubits pad with the rest
        chain_order += [q for q in range(n) if q not in chain_order]
        return {logical: physical for physical, logical in enumerate(chain_order)}
    # ------ general hardware: use Hungarian on spring embeddings ------
    import scipy.optimize

    # 2‑D spring embedding of both graphs
    pos_log = nx.spring_layout(inter_graph, seed=1, dim=2)
    pos_phy = nx.spring_layout(coupling_graph, seed=1, dim=2)
    # cost matrix = Euclidean distance in embedding
    C = np.zeros((n, n))
    for l in range(n):
        for p in range(n):
            C[l, p] = np.linalg.norm(pos_log.get(l, [0, 0]) - pos_phy[p])
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(C)
    return {int(l): int(p) for l, p in zip(row_ind, col_ind)}
