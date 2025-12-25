from typing import List, Dict

def compute_comm_mydclique(
    cliques: List[List[int]],
    clique_neighbors: List[List[int]],
) -> Dict[str, float]:
    """
    Recompute communication cost for myD-Clique.

    Assumptions (exactly matching your design):
    - Each clique has ONE aggregator
    - Non-aggregator nodes communicate ONLY with their aggregator
    - Inter-clique communication happens ONLY between aggregators
    - No full adjacency matrix is used
    """

    n_cliques = len(cliques)
    n_nodes = sum(len(c) for c in cliques)

    intra_msgs = 0
    inter_msgs = 0

    # -------- Intra-clique communication --------
    # per clique:
    # (k-1) uploads to aggregator
    # (k-1) broadcasts from aggregator
    for clique in cliques:
        k = len(clique)
        intra_msgs += 2 * (k - 1)

    # -------- Inter-clique communication --------
    # aggregator <-> aggregator only
    seen = set()
    for c, neighs in enumerate(clique_neighbors):
        for n in neighs:
            edge = tuple(sorted((c, n)))
            if edge in seen:
                continue
            seen.add(edge)
            inter_msgs += 2  # bidirectional

    total_msgs = intra_msgs + inter_msgs

    return {
        "n_nodes": n_nodes,
        "n_cliques": n_cliques,
        "intra_msgs_per_round": intra_msgs,
        "inter_msgs_per_round": inter_msgs,
        "total_msgs_per_round": total_msgs,
        "avg_msgs_per_node": total_msgs / n_nodes,
    }
