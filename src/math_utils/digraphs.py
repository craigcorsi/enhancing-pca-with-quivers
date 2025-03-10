import sys

sys.path.append("../")

import networkx as nx
import numpy as np
import scipy.linalg as linalg

from src.math_utils.linear_algebra import *


def get_initial_nodes(G):
    """
    Retrieves a list of the initial nodes of a digraph.

    Parameters:
        G: A NetworkX digraph

    Returns:
        A list of the nodes of G which have in-degree 0
    """
    nodes = list(G.nodes)
    nodes_with_indeg_0 = filter(lambda v: G.in_degree(v) == 0, nodes)
    return list(nodes_with_indeg_0)


def get_final_nodes(G):
    """
    Retrieves a list of the final nodes of a digraph.

    Parameters:
        G: A NetworkX digraph

    Returns:
        A list of the nodes of G which have out-degree 0
    """
    nodes = list(G.nodes)
    nodes_with_outdeg_0 = filter(lambda v: G.out_degree(v) == 0, nodes)
    return list(nodes_with_outdeg_0)


def augment_DAG_with_root(G):
    """
    Constructs a digraph by appending a root node to an existing digraph, if a root node does not exist.
    A root node is the unique node in a digraph with in-degree 0.

    Parameters:
        G: A NetworkX DiGraph. Optionally, G may have a quiver representation given by node attributes "dimension" and "basis",
            along with edge attribute "edge_map". In this case, if a new root node is appended, then "dimension", "basis", and
            "edge_map" attributes are constructed for the root node and its incident edges

    Returns:
        A NetworkX DiGraph obtained from adding a new root node "*" to G if G does not have a root node
    """
    if not nx.is_directed_acyclic_graph(G):
        print("The directed graph must be acyclic.")
        return

    # Get the initial nodes of G
    initial_nodes = get_initial_nodes(G)

    # If G already has a node with label '*', returns G without adding a root
    if "*" in list(G.nodes):
        return G

    # If G already has a unique initial node, returns G without adding a root
    if len(initial_nodes) <= 1:
        return G

    # Create a new root node "*" and an edge connecting the root to each initial node of G
    G1 = G.copy()
    G1.add_node("*")
    new_edge_list = [("*", v) for v in initial_nodes]
    G1.add_edges_from(new_edge_list)

    # Get quiver representation attributes, if they exist
    dimensions = nx.get_node_attributes(G1, "dimension")
    bases = nx.get_node_attributes(G1, "basis")
    edge_maps = nx.get_edge_attributes(G1, "edge_map")

    # If G has quiver representation attributes, construct vector space and linear maps for the new edges
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

    # Set attributes
    nx.set_node_attributes(G1, dimensions, "dimension")
    nx.set_node_attributes(G1, bases, "basis")
    nx.set_edge_attributes(G1, edge_maps, "edge_map")
    return G1


# Determines an ordering of nodes that is compatible with the path-induced partial order
def topological_sort(G):
    """
    Determines a topologically-sorted list of the nodes of an acyclic digraph.

    Parameters:
        G: A NetworkX digraph, which must not contain any directed cycles

    Returns:
        A list of nodes of G, ordered so that for any edge of G, the head of the edge precedes the tail of the edge
    """
    # Check whether G is acyclic
    if not nx.is_directed_acyclic_graph(G):
        print("The directed graph must be acyclic.")
        return

    H = G.copy()
    node_ordering = []

    # Recursively add initial nodes to the list and remove these nodes from the graph, until no nodes remain
    while len(list(H.nodes)) > 0:
        current_node_layer = get_initial_nodes(H)
        node_ordering.extend(current_node_layer)
        H.remove_nodes_from(current_node_layer)

    return node_ordering
