import networkx as nx
import numpy as np
import scipy.linalg as linalg

from linear_algebra_utils import *



def initial_nodes(G):
    nodes = list(G.nodes)
    nodes_with_indeg_0 = filter(lambda v: G.in_degree(v) == 0, nodes)
    return list(nodes_with_indeg_0)
    

# Given an acyclic graph G, construct a new graph with a root vertex, whose edges connect to the nodes in G with indegree 0
def augment_DAG_with_root(G):
    if not nx.is_directed_acyclic_graph(G):
        print("The directed graph must be acyclic.")
        return
    G1 = G.copy()
    
    # If G already has a unique initial node, does not construct a root
    head_nodes = initial_nodes(G1)
    if len(head_nodes) <= 1:
        return G1

    G1.add_node('*')
    arb_edge_list = [('*', v) for v in head_nodes]
    G1.add_edges_from(arb_edge_list)

    dimensions = nx.get_node_attributes(G,'dimension')
    bases = nx.get_node_attributes(G,'basis')
    rep_maps = nx.get_edge_attributes(G,'rep map')

    # If G has a representation, construct vector space and linear maps whose initial node is the new root
    if len(bases) > 0 and len(rep_maps) > 0:
        bases_in_sum = []
        
        # Collect bases of constituent spaces
        for k in range(len(head_nodes)):
            node = head_nodes[k]
            basis = bases[node]
            bases_in_sum.append(basis)

        # Construct basis of direct sum and projection maps
        direct_sum_basis, projection_maps = direct_sum(bases_in_sum)
    
        # Add new data to augmented graph
        dimensions['*'] = len(direct_sum_basis[0])
        bases['*'] = direct_sum_basis
        for k in range(len(head_nodes)):
            rep_maps[('*',head_nodes[k])] = projection_maps[k]
        
    nx.set_node_attributes(G1, dimensions, 'dimension')
    nx.set_node_attributes(G1, bases, 'basis')
    nx.set_edge_attributes(G1, rep_maps, 'rep map')    
        
    return G1




# Determines a spanning arborescence for an unweighted directed graph G - must be acyclic with a unique vertex of in-degree 0
def spanning_arborescence(G):
    if not (nx.is_directed_acyclic_graph(G) and len(initial_nodes(G)) == 1):
        print("The directed graph must be an arborescence.")
        return

    root = initial_nodes(G)[0]
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







