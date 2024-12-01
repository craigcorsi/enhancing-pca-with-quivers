import numpy as np
import scipy.linalg as linalg

from linear_algebra_utils import *
from digraph_algorithms import *




# Generates the flow spaces and flow maps for a quiver representation Q 
# Q must be a DiGraph with 'dimension', 'basis', and 'rep map' object
# Q must also have a unique root node
def generate_space_of_sections(Q):
    Q_dimensions = nx.get_node_attributes(Q, 'dimension')
    Q_bases = nx.get_node_attributes(Q, 'basis')
    Q_rep_maps = nx.get_edge_attributes(Q, 'rep map')
    if (len(Q_dimensions) == 0) or (len(Q_bases) == 0) or (len(Q_rep_maps) == 0):
        print("Please define a quiver representation (node attrs 'dimension' and 'basis' and edge attrs 'rep map').")
        return

    if len(get_initial_nodes(Q)) != 1:
        print("Please augment with a root node, see augment_DAG_with_root().")
        return

    # Get rep maps including maps from root node
    Q_rep_maps = nx.get_edge_attributes(Q, 'rep map')
    node_list = topological_sort(Q)

    flow_spaces = {}
    flow_maps = {}
    Q_nodes = [node for node in node_list if node != '*']

    root = node_list[0]
    flow_spaces[root] = nx.get_node_attributes(Q, 'basis')[root].copy()
    dim = len(flow_spaces[root])
    flow_maps[root] = np.eye(dim)
    node_list.remove(root)

    section_space_at_root = Q_bases[root]

    while len(node_list) > 0:
        n = node_list.pop(0)
        predecessors = list(Q.predecessors(n))
        predecessor_flow_spaces = {p: flow_spaces[p] for p in predecessors}
        predecessor_flow_maps = {p: flow_maps[p] for p in predecessors}

        flow_space_intersection = subspace_intersection(list(predecessor_flow_spaces.values()))
        
        path_maps_from_predecessors = [Q_rep_maps[(p,n)].dot(predecessor_flow_maps[p]) for p in predecessors]
        equalizer_at_n = equalizer_subspace(path_maps_from_predecessors, res_basis = flow_space_intersection)

        section_space_at_root = subspace_intersection([section_space_at_root, equalizer_at_n])
        
        flow_spaces[n] = equalizer_at_n.copy()
        flow_maps[n] = path_maps_from_predecessors[0]

    sections = []
    section_basis = section_space_at_root.copy()
    for b in section_basis:
        sections.append({node: flow_maps[node].dot(b) for node in Q_nodes})

    Q.graph['sections'] = sections
    nx.set_node_attributes(Q, flow_spaces, 'flow space')
    nx.set_node_attributes(Q, flow_maps, 'flow map')









# Convert basis vectors in the space of sections from dictionaries to np arrays
def orthogonalize_section_basis(section_basis, node_list):
    combined_sections = []

    for b in section_basis:
        print(b)
        combined_vec = []
        for node in node_list:
            combined_vec.append(b[node])
        combined_vec = np.concatenate(combined_vec)
        combined_sections.append(combined_vec)

    # In the case of 0-dimensional basis, return without normalizing
    if np.power(combined_sections[0],2).sum() < 0.00001:
        return combined_sections

    # Convert to orthonormal basis of column vectors
    combined_sections = np.concatenate(combined_sections, axis=1)
    combined_sections = linalg.orth(combined_sections)
    normalized_basis = [combined_sections[:,i] for i in range(len(combined_sections[0]))]
    normalized_basis = [normalized_basis[i][:,np.newaxis] for i in range(len(normalized_basis))]

    return normalized_basis


        









    
