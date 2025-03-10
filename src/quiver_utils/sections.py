import sys

sys.path.append("../")

from copy import deepcopy

import numpy as np
import scipy.linalg as linalg

import networkx as nx

import src.math_utils.linear_algebra as linalg_utils
import src.math_utils.digraphs as digraph_utils
from src.quiver_utils.elementwise_models import *


class QuiverRepSections:
    """
    Calculates a basis for the space of sections of a quiver representation, then projects a dataset onto
    the basis. Implements Algorithm 3 from p18 of https://arxiv.org/abs/2104.10666

    Parameters:
        node_structure (dict): Each node is associated with a list of column names arising from a Pandas DataFrame.
    """

    def __init__(self, node_structure):
        self.node_structure = node_structure
        self.qrep = None
        self.qrep_aug = None
        self.qrep_nodes = None
        self.has_0dim_sections = False

    def fit(self, qrep):
        """
        Calculates a basis for the space of sections for a quiver representation.

        Parameters:
            qrep: A NetworkX DiGraph with node attributes 'dimension' and 'basis', and edge attribute 'edge_map'

        Returns:
            None
        """
        self.qrep = deepcopy(qrep)
        self.qrep_aug = None
        self.qrep_nodes = list(self.qrep.nodes)

        # Input qrep must be a DiGraph with 'dimension', 'basis', and 'rep map' object
        dimensions = nx.get_node_attributes(self.qrep, "dimension")
        bases = nx.get_node_attributes(self.qrep, "basis")
        edge_maps = nx.get_edge_attributes(self.qrep, "edge_map")
        if (len(dimensions) == 0) or (len(bases) == 0) or (len(edge_maps) == 0):
            print(
                "Please define a quiver representation (node attrs 'dimension' and 'basis' and edge attr 'edge_map')."
            )
            return
        # Augments the quiver representation with a root node if one is not specified, ensuring the quiver is an
        # arborescence.
        self.qrep_aug = digraph_utils.augment_DAG_with_root(self.qrep)

        # Generate sections
        self.generate_space_of_sections()

        # Indicate whether the space of sections is 0-dimensional
        if self.qrep.graph["sections"][0].sum() ** 2 < 0.00001:
            self.has_0dim_sections = True

    def transform(self, X):
        """
        Projects a dataset onto the basis of sections calculated by self.fit, if the basis is nonzero.

        Parameters:
            X: Pandas DataFrame whose columns must include the column names contained in self.node_structure

        Returns:
            The projection of the data onto the basis of sections, if the basis is nonzero. Otherwise, returns an
            empty Pandas DataFrame of shape (len(X), 0).
        """
        # Handle case of 0-dimensional sections
        if self.has_0dim_sections:
            return pd.DataFrame(np.zeros(shape=(len(X), 0)))

        return self.project_onto_section_basis(X)

    def get_quiver_rep(self):
        """
        Retrieves the quiver representation.
        """
        return self.qrep

    def get_arborescence(self):
        """
        Retrieves the arborescence created during self.fit.
        """
        return self.qrep_aug

    def generate_space_of_sections(self, concatenate=True):
        """
        Generates the flow spaces, flow maps, and sections for a quiver representation. Implements Algorithm 3 from
        p18 of https://arxiv.org/abs/2104.10666

        Parameters:
            concatenate (bool): Set to True by default. When this is set to True, the output basis vectors will be
            NumPy arrays. When this is set to False, the output basis vectors will be formatted as dictionaries.

        Returns:
            The projection of the data onto the basis of sections, if the basis is nonzero. Otherwise, returns an
            empty Pandas DataFrame of shape (len(X), 0).
        """

        qrep = self.qrep
        qrep_aug = self.qrep_aug
        dimensions = nx.get_node_attributes(qrep_aug, "dimension")
        bases = nx.get_node_attributes(qrep_aug, "basis")
        edge_maps = nx.get_edge_attributes(qrep_aug, "edge_map")

        # Get topological sorting of nodes
        node_list = digraph_utils.topological_sort(qrep_aug)

        flow_spaces = {}
        flow_maps = {}

        # Compute flow spaces and flow maps for the root node of the arborescence
        root = node_list[0]
        flow_spaces[root] = nx.get_node_attributes(qrep_aug, "basis")[root].copy()
        flow_maps[root] = np.eye(len(flow_spaces[root]))
        section_space_at_root = bases[root]
        node_list.remove(root)

        while len(node_list) > 0:
            n = node_list.pop(0)
            predecessors = list(self.qrep_aug.predecessors(n))
            predecessor_flow_spaces = {p: flow_spaces[p] for p in predecessors}
            predecessor_flow_maps = {p: flow_maps[p] for p in predecessors}

            flow_space_intersection = linalg_utils.subspace_intersection(
                list(predecessor_flow_spaces.values())
            )

            path_maps_from_predecessors = [
                edge_maps[(p, n)].dot(predecessor_flow_maps[p]) for p in predecessors
            ]

            equalizer_at_n = linalg_utils.equalizer_subspace(
                path_maps_from_predecessors, res_basis=flow_space_intersection
            )

            section_space_at_root = linalg_utils.subspace_intersection(
                [section_space_at_root, equalizer_at_n]
            )

            flow_spaces[n] = equalizer_at_n.copy()
            flow_maps[n] = path_maps_from_predecessors[0]

        sections = []
        section_basis = section_space_at_root.copy()
        for b in section_basis:
            sections.append({node: flow_maps[node].dot(b) for node in self.qrep_nodes})

        nx.set_node_attributes(qrep, flow_spaces, "flow_space")
        nx.set_node_attributes(qrep, flow_maps, "flow_map")

        if concatenate:
            sections = self.concatenate_sections(sections)
            sections = self.normalize_sections(sections)
        qrep.graph["sections"] = sections

    def concatenate_sections(self, section_basis):
        """
        Convert basis vectors in the space of sections from dictionaries to NumPy arrays
        """
        concatenated_sections = []

        for b in section_basis:
            concatenated_vec = []
            for node in self.qrep_nodes:
                concatenated_vec.append(b[node])
            concatenated_vec = np.concatenate(concatenated_vec)
            concatenated_sections.append(concatenated_vec)
        return concatenated_sections

    def normalize_sections(self, section_basis):
        """
        Normalize basis vectors in the space of sections
        """
        # In the case of 0-dimensional basis, return without normalizing
        if np.power(section_basis[0], 2).sum() < 0.00001:
            return section_basis

        # Convert to orthonormal basis of column vectors
        concatenated_basis = np.concatenate(section_basis, axis=1)
        concatenated_basis = linalg.orth(concatenated_basis)
        normalized_basis = [
            concatenated_basis[:, i][:, np.newaxis]
            for i in range(len(concatenated_basis[0]))
        ]

        return normalized_basis

    def project_onto_section_basis(self, X):
        """
        Projects a dataset onto the basis of sections calculated by self.fit.

        Parameters:
            X: Pandas DataFrame passed to self.transform

        Returns:
            The projection of the data onto the basis of sections.
        """
        qrep = self.qrep
        display_name = (
            self.qrep.graph["name"]
            if len(self.qrep.graph["name"]) > 0
            else self.qrep.graph["id"]
        )

        # Get the cols of the data that belong to the nodes in qrep
        nodes = list(qrep.nodes)
        X_qrep = pd.concat([X[self.node_structure[node]] for node in nodes], axis=1)

        # Get the basis of sections
        sections = qrep.graph["sections"]

        X_qrep_proj = [
            linalg_utils.project_onto_subspace_in_basis(X_qrep.iloc[i], sections)
            for i in range(len(X_qrep))
        ]
        X_qrep_proj = pd.concat(X_qrep_proj, axis=1).T

        # Dimension reduction for the projected data: only take principal components comprising the top 95%
        # of explained variance ratio
        pca = TopPrincipalComponents(use_threshold=True, label_name=display_name)
        pca.fit(X_qrep_proj)
        X_qrep_proj = pca.transform(X_qrep_proj)

        return X_qrep_proj


class QuiverRepSectionsMulti:
    """
    Projects a dataset onto the space of sections for multiple quiver representations.

    Parameters:
    node_structure (dict): Each node is associated with a list of column names arising from a Pandas DataFrame.
    """

    def __init__(self, node_structure):
        self.node_structure = node_structure
        self.qrep_list = []
        self.model_list = []

    def fit(self, qrep_list):
        """
        Calculates a list of bases for the space of sections of multiple quiver representations.

        Parameters:
            qrep_list: A list of NetworkX DiGraphs with node attributes 'dimension' and 'basis', and edge
                attribute 'edge_map'

        Returns:
            None
        """
        self.model_list = [
            QuiverRepSections(self.node_structure) for i in range(len(qrep_list))
        ]
        for i in range(len(self.model_list)):
            self.model_list[i].fit(qrep_list[i])
        # Update qrep_list after calculating sections
        self.qrep_list = [model.qrep for model in self.model_list]

    def transform(self, X):
        """
        Projects a dataset onto the bases of sections calculated by self.fit

        Parameters:
            X: Pandas DataFrame whose columns must include the column names contained in self.node_structure

        Returns:
            The projection of the data onto the basis of sections.
        """
        X_quiver_projections = []
        for i in range(len(self.model_list)):
            X_proj = self.model_list[i].transform(X)
            X_quiver_projections.append(X_proj)
        X_quiver_invariant = pd.concat(X_quiver_projections, axis=1)
        return X_quiver_invariant

    def get_quiver_reps(self):
        """
        Retrieves the list of quiver representations.
        """
        return self.qrep_list
