import sys

sys.path.append("../")

import networkx as nx
import numpy as np
import scipy.linalg as linalg

from src.math_utils.linear_algebra import *


def get_initial_nodes(G):
    nodes = list(G.nodes)
    nodes_with_indeg_0 = filter(lambda v: G.in_degree(v) == 0, nodes)
    return list(nodes_with_indeg_0)


def get_final_nodes(G):
    nodes = list(G.nodes)
    nodes_with_outdeg_0 = filter(lambda v: G.out_degree(v) == 0, nodes)
    return list(nodes_with_outdeg_0)


# Given an acyclic graph, construct a new graph with a root vertex, whose edges connect to the nodes having indegree 0
def augment_DAG_with_root(G):
    if not nx.is_directed_acyclic_graph(G):
        print("The directed graph must be acyclic.")
        return

    # If G already has a node with label '*', does not construct a root
    if "*" in list(G.nodes):
        return G

    # If G already has a unique initial node, does not construct a root
    initial_nodes = get_initial_nodes(G)
    if len(initial_nodes) <= 1:
        return G

    G1 = G.copy()

    G1.add_node("*")
    arb_edge_list = [("*", v) for v in initial_nodes]
    G1.add_edges_from(arb_edge_list)

    dimensions = nx.get_node_attributes(G1, "dimension")
    bases = nx.get_node_attributes(G1, "basis")
    edge_maps = nx.get_edge_attributes(G1, "edge_map")

    # If G has a representation, construct vector space and linear maps whose initial node is the new root
    if len(bases) > 0 and len(edge_maps) > 0:
        bases_in_sum = []

        # Collect bases of constituent spaces
        for k in range(len(initial_nodes)):
            node = initial_nodes[k]
            basis = bases[node]
            bases_in_sum.append(basis)

        # Construct basis of direct sum and projection maps
        direct_sum_basis, projection_maps = direct_sum(bases_in_sum)

        # Add new data to augmented graph
        dimensions["*"] = len(direct_sum_basis[0])
        bases["*"] = direct_sum_basis
        for k in range(len(initial_nodes)):
            edge_maps[("*", initial_nodes[k])] = projection_maps[k]

    nx.set_node_attributes(G1, dimensions, "dimension")
    nx.set_node_attributes(G1, bases, "basis")
    nx.set_edge_attributes(G1, edge_maps, "edge_map")
    return G1


# Determines an ordering of nodes that is compatible with the path-induced partial order
def topological_sort(G):
    if not nx.is_directed_acyclic_graph(G):
        print("The directed graph must be acyclic.")
        return

    H = G.copy()
    node_ordering = []

    while len(list(H.nodes)) > 0:
        current_node_layer = get_initial_nodes(H)
        node_ordering.extend(current_node_layer)
        H.remove_nodes_from(current_node_layer)

    return node_ordering
