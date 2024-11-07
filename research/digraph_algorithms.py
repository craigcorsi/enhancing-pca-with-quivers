import networkx as nx
import numpy as np
import scipy.linalg as linalg

from linear_algebra_utils import *



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
    if '*' in list(G.nodes):
        return
    
    # If G already has a unique initial node, does not construct a root
    initial_nodes = get_initial_nodes(G)
    if len(initial_nodes) <= 1:
        return

    G1 = G.copy()

    G1.add_node('*')
    arb_edge_list = [('*', v) for v in initial_nodes]
    G1.add_edges_from(arb_edge_list)

    dimensions = nx.get_node_attributes(G1,'dimension')
    bases = nx.get_node_attributes(G1,'basis')
    rep_maps = nx.get_edge_attributes(G1,'rep map')

    # If G has a representation, construct vector space and linear maps whose initial node is the new root
    if len(bases) > 0 and len(rep_maps) > 0:
        bases_in_sum = []
        
        # Collect bases of constituent spaces
        for k in range(len(initial_nodes)):
            node = initial_nodes[k]
            basis = bases[node]
            bases_in_sum.append(basis)

        # Construct basis of direct sum and projection maps
        direct_sum_basis, projection_maps = direct_sum(bases_in_sum)
    
        # Add new data to augmented graph
        dimensions['*'] = len(direct_sum_basis[0])
        bases['*'] = direct_sum_basis
        for k in range(len(initial_nodes)):
            rep_maps[('*',initial_nodes[k])] = projection_maps[k]
        
    nx.set_node_attributes(G1, dimensions, 'dimension')
    nx.set_node_attributes(G1, bases, 'basis')
    nx.set_edge_attributes(G1, rep_maps, 'rep map')
    return G1






# Determines an ordering of nodes that is compatible with the path-induced partial order
def topological_sort(G):
    if not nx.is_directed_acyclic_graph(G):
        print("The directed graph must be acyclic.")
        return

    node_ordering = []
    current_node_layer = get_initial_nodes(G)
    undiscovered_nodes = list(G.nodes)
    for s in current_node_layer:
        node_ordering.append(s)
        undiscovered_nodes.remove(s)

    while len(current_node_layer) > 0:
        # Construct the next node layer using only undiscovered neighbors of the nodes in current_node_layer
        new_nodes = []
        for n in current_node_layer:
            neighbors = list(nx.neighbors(G,n))
            neighbors = list(filter(lambda s: s in undiscovered_nodes, neighbors))

            for s in neighbors:
                new_nodes.append(s)
                node_ordering.append(s)
                undiscovered_nodes.remove(s)
        current_node_layer = new_nodes.copy()

    return node_ordering  









# Determines a spanning arborescence for an unweighted directed graph G - must be acyclic with a unique vertex of in-degree 0
def spanning_arborescence(G):
    if not (nx.is_directed_acyclic_graph(G) and len(get_initial_nodes(G)) == 1):
        print("The directed graph must be an arborescence.")
        return

    root = get_initial_nodes(G)[0]
    undiscovered_nodes = list(G.nodes)
    undiscovered_nodes.remove(root)
    current_node_layer = [root]
    spanning_arb_edges = []

    while len(current_node_layer) > 0:
        # Construct the next node layer using only undiscovered neighbors of the nodes in current_node_layer
        new_nodes = []
        for n in current_node_layer:
            neighbors = list(nx.neighbors(G,n))
            neighbors = list(filter(lambda s: s in undiscovered_nodes, neighbors))

            for s in neighbors:
                new_nodes.append(s)
                spanning_arb_edges.append((n,s))
                undiscovered_nodes.remove(s)
        current_node_layer = new_nodes.copy()

    Tp = G.edge_subgraph(spanning_arb_edges)
    return Tp   







