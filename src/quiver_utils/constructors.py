import sys

sys.path.append("../")

from copy import deepcopy
import json

import pandas as pd
import numpy as np

import networkx as nx

from src.math_utils.linear_algebra import *

from scipy.stats import gaussian_kde


class DataQuiverRepConstructor:
    def __init__(
        self,
        node_structure,
        edge_maps,
        default_node_limit=None,
        default_build_direction=None,
    ):
        self.node_structure = node_structure
        self.edge_maps = edge_maps
        self.nodes = list(node_structure.keys())
        self.num_nodes = len(self.nodes)

        # Set default node limit
        if default_node_limit:
            self.default_node_limit = min([default_node_limit, self.num_nodes])
        else:
            self.default_node_limit = self.num_nodes

        # Set default build direction
        self.default_build_direction = default_build_direction or "incoming-outgoing"

        # Generate a unique id for each quiver representation
        self._qrep_id_counter = 0

        # Temporary values for the dataset used to construct a quiver representation
        self._initial_tempvals_data = {
            # Dataset used to construct quiver
            "data": None,
            # Number of points in the dataset
            "num_pts": None,
            # The index of the reference node
            "ref_index": None,
            # The slice of the data at the reference node's columns
            "ref_node_data": None,
        }

        # Temporary values when recursively adding a new edge to a quiver
        self._initial_tempvals_addedge = {
            # Node added to the quiver most recently
            "last_added_node": None,
            # New edges that can be added to the quiver
            "new_edges": [],
            # All possible edges that can be added to the quiver
            "next_edge_possibilities": [],
            # Edges selected to be added to the quiver
            "quiver_edges": [],
            # Nodes that have not been added to the quiver
            "unused_nodes": [],
        }

        # Temporary values describing digraph attributes at the graph-, node-, and edge-level
        # These are to be passed to a NetworkX Quiver with the exception the attributes in self._skipped_attrs
        self._initial_tempvals_qrep = {
            # Graph attributes
            "graph": {
                # Direction in which edges are added to the quiver, relative to the ref node
                "build_direction": None,
                # Filename
                "filename": None,
                # Unique ID
                "id": None,
                # Name of quiver representation
                "name": None,
                # The maximum number of nodes to be added to the quiver
                "node_limit": None,
                # To be updated as nodes are added, accessing nodes by layer
                "nodes_by_layer": {
                    d: [] for d in range(-self.num_nodes, self.num_nodes + 1)
                },
                # The reference node placed at layer 0
                "ref_node": None,
            },
            # Node attributes
            "nodes": {
                node: {
                    "direction": None,
                    "entropy": None,
                    "layer": None,
                    "ref_path_map": None,
                }
                for node in self.nodes
            },
            # Edge attributes
            "edges": {
                edge: {
                    "direction": None,
                    "edge": None,
                    "edge_map": None,
                    "entropy": None,
                    "head": None,
                    "path_head": None,
                    "ref_path_map": None,
                    "path_tail": None,
                    "residuals": None,
                    "tail": None,
                }
                for edge in []
            },
        }

        # Skip these attributes when constructing the NetworkX digraph
        self._skipped_attrs = ["entropy", "residuals"]

        # Initialize temporary values
        self._tempvals_addedge = None
        self._tempvals_data = None
        self._tempvals_qrep = None
        self._reset_temp_values()

        # The most recently constructed quiver representation
        self.most_recent_qrep = None

        # A running list of quiver representations
        self.saved_qreps = []

    def fit_one_quiver_rep(
        self,
        data,
        ref_index=None,
        ref_node=None,
        build_direction=None,
        node_limit=None,
        qrep_name=None,
        save_qrep=True,
    ):

        # Validate ref_index and ref_node
        if not self._validate_ref_node(ref_index, ref_node):
            return
        ref_index, ref_node = self._fill_values_ref_node(ref_index, ref_node)

        # Reset temporary values
        self._reset_temp_values()

        # Set id, name, and filename for the quiver rep
        qrep_id = f"Q{self._qrep_id_counter:03}"
        qrep_name = qrep_name or ""
        self._tempvals_qrep["graph"]["id"] = qrep_id
        self._qrep_id_counter += 1
        self._tempvals_qrep["graph"]["name"] = qrep_name
        self._tempvals_qrep["graph"]["filename"] = (
            f"{qrep_id}_{qrep_name}" if len(qrep_name) > 0 else qrep_id
        )

        # Print in-progress message
        qrep_name_display = (
            qrep_name if qrep_name else self._tempvals_qrep["graph"]["id"]
        )
        print(f"Generating the quiver rep {qrep_name_display}!")

        # Set parameters that are not specified
        build_direction, node_limit = self._fill_values_one_qrep_params(
            build_direction, node_limit
        )

        # Store data for the ref node
        ref_node_data = data[self.node_structure[ref_node]]

        # Record dataset-related info
        self._tempvals_data.update(
            {
                "data": data,
                "num_pts": len(data),
                "ref_index": ref_index,
                "ref_node_data": ref_node_data,
            }
        )
        # Record quiver rep info for the reference node
        self._tempvals_qrep["graph"]["nodes_by_layer"][0].append(ref_node)
        self._tempvals_qrep["nodes"][ref_node].update(
            {
                "direction": "ref",
                "entropy": 0.0,
                "layer": 0,
                "ref_path_map": np.eye(len(self.node_structure[ref_node])),
            }
        )
        self._tempvals_qrep["graph"].update(
            {
                "build_direction": build_direction,
                "node_limit": node_limit or self.default_node_limit,
                "ref_node": ref_node,
            }
        )
        # Update temporary values for the reference node
        self._tempvals_addedge["last_added_node"] = ref_node
        self._tempvals_addedge["unused_nodes"] = [
            node for node in self.nodes if node != ref_node
        ]

        # Select edges, then construct the quiver representation
        for i in range(node_limit - 1):
            self._attach_next_edge()
        self.most_recent_qrep = self._construct_quiver_representation(
            qrep_name=qrep_name
        )

        # Save quiver rep
        if save_qrep:
            self.saved_qreps.append(self.most_recent_qrep)

        return self.most_recent_qrep

    def fit_transform(
        self,
        data,
        build_direction=None,
        build_directions: list = [],
        node_limit=None,
        node_limits: list = [],
        qrep_names: list = [],
        ref_indices: list = [],
        ref_nodes: list = [],
        save_qrep=True,
    ):

        # Validate ref_indices and ref_nodes
        if not self._validate_multiple_ref_nodes(ref_indices, ref_nodes):
            return
        # Set values that are not specified
        ref_indices, ref_nodes = self._fill_values_multiple_ref_nodes(
            ref_indices, ref_nodes
        )

        # Determine the number of quivers to be constructed
        num_qreps = len(ref_indices)

        # Validate all other quiver rep params
        if not self._validate_multiple_qrep_params(
            num_qreps, build_directions, node_limits, qrep_names
        ):
            return
        # Set parameters that are not specified
        build_direction, node_limit, build_directions, node_limits, qrep_names = (
            self._fill_values_multiple_qrep_params(
                num_qreps,
                build_direction,
                node_limit,
                build_directions,
                node_limits,
                qrep_names,
            )
        )

        # Construct quiver representations
        quiver_reps = []
        for i in range(num_qreps):
            qrep = self.fit_one_quiver_rep(
                data,
                build_direction=build_directions[i],
                node_limit=node_limits[i],
                qrep_name=qrep_names[i],
                ref_index=ref_indices[i],
                ref_node=ref_nodes[i],
                save_qrep=save_qrep,
            )
            quiver_reps.append(qrep)
        return quiver_reps

    def get_most_recent_qrep(self):
        return self.most_recent_qrep

    def get_saved_qreps(self):
        return self.saved_qreps

    def delete_saved_qreps(self):
        self.saved_qreps = []

    ########## Recursive step to add new edges

    def _attach_next_edge(self):
        self._get_new_edges()
        self._add_best_edge()
        self._remove_unusable_edges()

    # Choose a new edge for the quiver by first adding eligible edges incident to the last-added node
    def _get_new_edges(self):
        # Construct all valid edges incident to the last added node
        direction = self._tempvals_qrep["nodes"][
            self._tempvals_addedge["last_added_node"]
        ]["direction"]
        # For the reference node of an incoming-outgoing quiver, construct incoming and outgoing edges
        if (
            direction == "ref"
            and self._tempvals_qrep["graph"]["build_direction"] == "incoming-outgoing"
        ):
            edges_incoming = self._enumerate_edges("incoming")
            edges_outgoing = self._enumerate_edges("outgoing")
            self._tempvals_addedge["new_edges"] = self._construct_edge_objects(
                edges_incoming, "incoming"
            ) + self._construct_edge_objects(edges_outgoing, "outgoing")
        # Otherwise, the new edges will be either all incoming or all outgoing
        else:
            if direction == "ref":
                direction = self._tempvals_qrep["graph"]["build_direction"]
            edges = self._enumerate_edges(direction)
            self._tempvals_addedge["new_edges"] = self._construct_edge_objects(
                edges, direction
            )

        # Record path maps
        for new_edge in self._tempvals_addedge["new_edges"]:
            new_edge["ref_path_map"] = self._compose_ref_path_map(new_edge)

        # Delete edges whose path map is approximately the zero map
        self._tempvals_addedge["new_edges"] = [
            new_edge
            for new_edge in self._tempvals_addedge["new_edges"]
            if not self._has_vanishing_ref_path_map(new_edge)
        ]

        # Calculate residuals (data_actual - data_predicted) and their entropy scores
        for new_edge in self._tempvals_addedge["new_edges"]:
            data_actual = self._node_data_actual(new_edge)
            data_predicted = self._node_data_predicted(new_edge)
            new_edge["residuals"] = data_actual - data_predicted
            new_edge["entropy"] = self._entropy_score(new_edge)

        # Add new edges to the list of next edge possibilities
        self._tempvals_addedge["next_edge_possibilities"].extend(
            self._tempvals_addedge["new_edges"]
        )

    def _add_best_edge(self):
        # Determine the edge with the lowest entropy score and the new node incident to it
        _entropy_scores = [
            e["entropy"] for e in self._tempvals_addedge["next_edge_possibilities"]
        ]
        new_edge_index = _entropy_scores.index(min(_entropy_scores))
        new_edge = self._tempvals_addedge["next_edge_possibilities"][new_edge_index]
        previous_node = self._previous_node_from_new_edge(new_edge)
        new_node = self._new_node_from_new_edge(new_edge)

        new_layer = self._layer_of_new_node(new_edge)
        new_direction = new_edge["direction"]
        new_entropy = new_edge["entropy"]
        new_ref_path_map = new_edge["ref_path_map"]

        # Update temporary values with new edge and new node
        self._tempvals_addedge["last_added_node"] = str(new_node)
        self._tempvals_addedge["quiver_edges"].append(new_edge)
        self._tempvals_qrep["graph"]["nodes_by_layer"][new_layer].append(new_node)
        self._tempvals_qrep["nodes"][new_node].update(
            {
                "direction": new_direction,
                "entropy": new_entropy,
                "layer": new_layer,
                "ref_path_map": new_ref_path_map,
            }
        )
        self._tempvals_qrep["edges"][new_edge["edge"]] = new_edge

        print(f"{new_edge['edge']} to layer {new_layer}")

    def _remove_unusable_edges(self):
        # Remove new node from list of unused nodes
        new_node = self._tempvals_addedge["last_added_node"]
        self._tempvals_addedge["unused_nodes"] = [
            node for node in self._tempvals_addedge["unused_nodes"] if node != new_node
        ]
        # Remove all edges incident to the new node from the list of next edge possibilities
        self._tempvals_addedge["next_edge_possibilities"] = [
            e
            for e in self._tempvals_addedge["next_edge_possibilities"]
            if (e["head"] != new_node and e["tail"] != new_node)
        ]

    ########## Construct quiver representation from complete list of selected edges

    def _construct_quiver_representation(self, qrep_name=None):
        qrep_name = qrep_name or "Q"
        qrep = nx.DiGraph()
        qrep.graph.update({"name": qrep_name})
        qrep.graph.update(self._tempvals_qrep["graph"])
        # Add edges and edge attributes
        for edge in self._tempvals_addedge["quiver_edges"]:
            head = edge["head"]
            tail = edge["tail"]
            qrep.add_edge(tail, head)
            edge_attrs = self._filter_df_attrs(
                self._tempvals_qrep["edges"][edge["edge"]]
            )
            qrep.edges[tail, head].update(edge_attrs)

        # Add node attributes for dimension and basis
        nodes = list(qrep.nodes)
        dimensions = {node: len(self.node_structure[node]) for node in nodes}
        bases = construct_standard_bases(dimensions)
        nx.set_node_attributes(qrep, dimensions, "dimension")
        nx.set_node_attributes(qrep, bases, "basis")
        # Add remaining node attributes
        for node in nodes:
            node_attrs = self._filter_df_attrs(self._tempvals_qrep["nodes"][node])
            qrep.nodes[node].update(node_attrs)

        return qrep

    ########## Auxiliary math and retrieval functions

    def _reset_temp_values(self):
        self._tempvals_data = deepcopy(self._initial_tempvals_data)
        self._tempvals_addedge = deepcopy(self._initial_tempvals_addedge)
        self._tempvals_qrep = deepcopy(self._initial_tempvals_qrep)

    def _filter_df_attrs(self, attr_dict):
        return {
            key: val for key, val in attr_dict.items() if key not in self._skipped_attrs
        }

    def _negative_log_pdf(self, x, pdf):
        pdfx = pdf.evaluate(x)
        if np.abs(pdfx) < 1e-20:
            return 0
        else:
            return -np.log(pdfx)[0]

    def _entropy_score(self, new_edge):
        path_head = new_edge["path_head"]
        path_tail = new_edge["path_tail"]
        residuals = new_edge["residuals"]

        # Estimate probability density function of the residuals using a Gaussian KDE
        pdf = gaussian_kde(residuals.T)
        sigma = pdf.factor

        # Scale to correct for dimensions of domain/codomain of path maps
        d1 = len(self.node_structure[path_tail])
        d2 = len(self.node_structure[path_head])
        # This is an experimental guess
        scaling_factor = 1 / (d1 * d2)

        # Sample from the residuals' probability density function to approximate the Shannon differential entropy
        num_samples = 100
        samples = []
        for i in range(num_samples):
            j = np.random.randint(self._tempvals_data["num_pts"])
            pj = residuals.iloc[j].to_numpy()
            xj = np.random.multivariate_normal(pj, sigma * np.eye(len(pj)))
            samples.append(self._negative_log_pdf(xj, pdf))
        return scaling_factor * np.array(samples).mean()

    def _compose_ref_path_map(self, new_edge):
        direction = new_edge["direction"]
        head = new_edge["head"]
        tail = new_edge["tail"]
        if direction == "incoming":
            closer_node = head
            farther_node = tail
        elif direction == "outgoing":
            closer_node = tail
            farther_node = head
        closer_node_ref_path_map = self._tempvals_qrep["nodes"][closer_node][
            "ref_path_map"
        ]
        new_edge_map = self.edge_maps[tail][head]
        if direction == "incoming":
            ref_path_map = closer_node_ref_path_map.dot(new_edge_map)
        elif direction == "outgoing":
            ref_path_map = new_edge_map.dot(closer_node_ref_path_map)
        return ref_path_map

    def _has_vanishing_ref_path_map(self, new_edge):
        ref_path_map = new_edge["ref_path_map"]
        return np.abs(ref_path_map).sum() < 0.01

    def _node_data_actual(self, new_edge):
        direction = new_edge["direction"]
        if direction == "incoming":
            return self._tempvals_data["ref_node_data"].copy()
        elif direction == "outgoing":
            head = new_edge["head"]
            return self._tempvals_data["data"][self.node_structure[head]]

    def _node_data_predicted(self, new_edge):
        direction = new_edge["direction"]
        ref_path_map = new_edge["ref_path_map"]
        if direction == "incoming":
            tail = new_edge["tail"]
            data_to_map = self._tempvals_data["data"][self.node_structure[tail]]
        elif direction == "outgoing":
            data_to_map = self._tempvals_data["ref_node_data"].copy()
        predicted = data_to_map.dot(ref_path_map.T)
        if direction == "incoming":
            predicted.columns = self.node_structure[
                self._tempvals_qrep["graph"]["ref_node"]
            ]
        elif direction == "outgoing":
            predicted.columns = self.node_structure[new_edge["head"]]
        return predicted

    def _enumerate_edges(self, direction):
        if direction == "incoming":
            return [
                (node, self._tempvals_addedge["last_added_node"])
                for node in self._tempvals_addedge["unused_nodes"]
            ]
        elif direction == "outgoing":
            return [
                (self._tempvals_addedge["last_added_node"], node)
                for node in self._tempvals_addedge["unused_nodes"]
            ]

    def _get_path_tail(self, edge, direction):
        if direction == "incoming":
            return edge[0]
        elif direction == "outgoing":
            return self._tempvals_qrep["graph"]["ref_node"]

    def _get_path_head(self, edge, direction):
        if direction == "incoming":
            return self._tempvals_qrep["graph"]["ref_node"]
        elif direction == "outgoing":
            return edge[1]

    def _construct_edge_objects(self, edges, direction):
        return [
            {
                "direction": direction,
                "edge": edge,
                "tail": edge[0],
                "head": edge[1],
                "path_tail": self._get_path_tail(edge, direction),
                "path_head": self._get_path_head(edge, direction),
                "edge_map": self.edge_maps[edge[0]][edge[1]],
            }
            for edge in edges
        ]

    def _layer_of_new_node(self, new_edge):
        direction = new_edge["direction"]
        if direction == "incoming":
            closer_node = new_edge["head"]
            return self._tempvals_qrep["nodes"][closer_node]["layer"] - 1
        elif direction == "outgoing":
            closer_node = new_edge["tail"]
            return self._tempvals_qrep["nodes"][closer_node]["layer"] + 1

    def _new_node_from_new_edge(self, new_edge):
        direction = new_edge["direction"]
        if direction == "incoming":
            return new_edge["tail"]
        elif direction == "outgoing":
            return new_edge["head"]

    def _previous_node_from_new_edge(self, new_edge):
        direction = new_edge["direction"]
        if direction == "incoming":
            return new_edge["head"]
        elif direction == "outgoing":
            return new_edge["tail"]

    ########## Validation of parameters

    def _validate_ref_node(self, ref_index, ref_node):
        # Check whether either a ref_index or a ref_node is specified
        if not ref_index and not ref_node:
            print("Please specify either a ref_node or a ref_index.")
            return False
        # Check if both are specified but do not match
        if ref_index and ref_node and ref_node != self.nodes[ref_index]:
            print(
                "The specified ref_index and ref_node do not match. Only one should be specified."
            )
            return False
        return True

    def _validate_multiple_ref_nodes(self, ref_indices, ref_nodes):
        # If both ref_indices and ref_nodes are specified, check whether they match
        if len(ref_indices) > 0 and len(ref_nodes) > 0:
            if len(ref_indices) != len(ref_nodes):
                print(
                    "Invalid parameters. If ref_indices and ref_nodes are both specified then they must match."
                )
                return False
            else:
                for i in range(len(ref_indices)):
                    if self.nodes[ref_indices[i]] != ref_nodes[i]:
                        print(
                            "Invalid parameters. If ref_indices and ref_nodes are both specified then they must match."
                        )
                        return False
        return True

    def _validate_multiple_qrep_params(
        self, num_qreps, build_directions, node_limits, qrep_names
    ):
        # Check for validity of parameters
        if not (
            (
                len(build_directions) in [0, num_qreps]
                and len(node_limits) in [0, num_qreps]
                and len(qrep_names) in [0, num_qreps]
            )
        ):
            print(
                "Invalid parameters. The lists build_directions, node_limits, or qrep_names must have length zero or the same length as the number of ref_nodes (or ref_indices)."
            )
            return False
        return True

    def _fill_values_ref_node(self, ref_index, ref_node):
        ref_index = ref_index or self.nodes.index(ref_node)
        ref_node = ref_node or self.nodes[ref_index]
        return ref_index, ref_node

    def _fill_values_one_qrep_params(self, build_direction, node_limit):
        if build_direction == "random":
            build_direction = np.random.choice(
                ["incoming", "outgoing", "incoming-outgoing"]
            )
        else:
            build_direction = build_direction or self.default_build_direction
        node_limit = node_limit or self.default_node_limit
        return build_direction, node_limit

    def _fill_values_multiple_ref_nodes(self, ref_indices, ref_nodes):
        if len(ref_indices) == 0 and len(ref_nodes) == 0:
            ref_indices = np.arange(self.num_nodes)
            ref_nodes = self.nodes
        elif len(ref_indices) == 0:
            ref_indices = [self.nodes.index(ref_node) for ref_node in ref_nodes]
        elif len(ref_nodes) == 0:
            ref_nodes = [self.nodes[ref_index] for ref_index in ref_indices]
        return ref_indices, ref_nodes

    def _fill_values_multiple_qrep_params(
        self,
        num_qreps,
        build_direction,
        node_limit,
        build_directions,
        node_limits,
        qrep_names,
    ):
        build_direction = build_direction or self.default_build_direction
        node_limit = node_limit or self.default_node_limit
        if len(build_directions) == 0:
            build_directions = [build_direction] * num_qreps
        if len(node_limits) == 0:
            node_limits = [node_limit] * num_qreps
        if len(qrep_names) == 0:
            qrep_names = [None for i in range(num_qreps)]
        return build_direction, node_limit, build_directions, node_limits, qrep_names
