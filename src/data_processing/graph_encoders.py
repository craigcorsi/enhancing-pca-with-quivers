import json
from json import JSONEncoder, JSONDecoder

import pandas as pd
import numpy as np

import networkx as nx
from networkx.readwrite import json_graph


def decode_edge_map_collection(edge_maps_from_file):
    """
    Custom JSON decoding for EdgeMapRegression objects.

    Parameters:
        edge_maps_from_file: a dict-of-dicts whose values can be passed to numpy.array

    Returns:
        The dict-of-dicts obtained from edge_maps_from_file by converting each value from a list-of-lists to a NumPy array
    """
    maps_as_dict = edge_maps_from_file.copy()
    node_list = maps_as_dict.keys()
    for head in node_list:
        tails = maps_as_dict[head].keys()
        for tail in tails:
            maps_as_dict[head][tail] = np.array(maps_as_dict[head][tail])
    return maps_as_dict


def encode_JSON_quiver_rep_arrays(Q_as_dict):
    """
    Custom JSON encoding for NetworkX DiGraphs containing NumPy arrays.

    Parameters:
        Q_as_dict: the output of networkx.readwrite.json_graph.node_link_data

    Returns:
        The object obtained from Q_as_dict by converting each (graph, node, or edge) attributes of type (numpy.ndarray or
            list of numpy.ndarray objects) to lists. New graph attributes are created indicating which attributes are to
            be decoded
    """
    Q_to_file = Q_as_dict.copy()

    # Determine all graph attributes whose data type is either a numpy array or a list of numpy arrays and encode as lists
    nparray_graph_attributes = []
    list_of_nparrays_graph_attributes = []
    if len(Q_to_file["graph"]) > 0:
        graph_keys = Q_to_file["graph"].keys()
        for key in graph_keys:
            if isinstance(Q_to_file["graph"][key], np.ndarray):
                Q_to_file["graph"][key] = Q_to_file["graph"][key].tolist()
                np_array_attributes.append(key)
            if (
                isinstance(Q_to_file["graph"][key], list)
                and len(Q_to_file["graph"][key]) > 0
            ):
                b = Q_to_file["graph"][key][0]
                if isinstance(b, np.ndarray):
                    array_list_as_list_of_lists = [
                        b.tolist() for b in Q_to_file["graph"][key]
                    ]
                    Q_to_file["graph"][key] = array_list_as_list_of_lists
                    list_of_nparrays_graph_attributes.append(key)

    # Determine all node and edge attributes whose data type is either a numpy array or a list of numpy arrays
    nparray_node_attributes = []
    list_of_nparrays_node_attributes = []
    nparray_edge_attributes = []
    list_of_nparrays_edge_attributes = []

    node_keys = Q_to_file["nodes"][0].keys()
    for key in node_keys:
        if isinstance(Q_to_file["nodes"][0][key], np.ndarray):
            nparray_node_attributes.append(key)
        if (
            isinstance(Q_to_file["nodes"][0][key], list)
            and len(Q_to_file["nodes"][0][key]) > 0
        ):
            b = Q_to_file["nodes"][0][key][0]
            if isinstance(b, np.ndarray):
                list_of_nparrays_node_attributes.append(key)
    edge_keys = Q_to_file["edges"][0].keys()
    for key in edge_keys:
        if isinstance(Q_to_file["edges"][0][key], np.ndarray):
            nparray_edge_attributes.append(key)
        if (
            isinstance(Q_to_file["edges"][0][key], list)
            and len(Q_to_file["edges"][0][key]) > 0
        ):
            b = Q_to_file["edges"][0][key][0]
            if isinstance(b, np.ndarray):
                list_of_nparrays_edges_attributes.append(key)

    # Now iterate through every node and edge and encode all numpy arrays as lists
    for key in nparray_node_attributes:
        for i in range(len(Q_to_file["nodes"])):
            Q_to_file["nodes"][i][key] = Q_to_file["nodes"][i][key].tolist()
    for key in list_of_nparrays_node_attributes:
        for i in range(len(Q_to_file["nodes"])):
            array_list = Q_to_file["nodes"][i][key]
            array_list_as_list_of_lists = [b.tolist() for b in array_list]
            Q_to_file["nodes"][i][key] = array_list_as_list_of_lists

    for key in nparray_edge_attributes:
        for i in range(len(Q_to_file["edges"])):
            Q_to_file["edges"][i][key] = Q_to_file["edges"][i][key].tolist()
    for key in list_of_nparrays_edge_attributes:
        for i in range(len(Q_to_file["edges"])):
            array_list = Q_to_file["edges"][i][key]
            array_list_as_list_of_lists = [b.tolist() for b in array_list]
            Q_as_dict["edges"][i][key] = array_list_as_list_of_lists

    # Save nparray attributes for the decoder
    Q_to_file["graph"]["nparray_graph_attributes"] = nparray_graph_attributes
    Q_to_file["graph"][
        "list_of_nparrays_graph_attributes"
    ] = list_of_nparrays_graph_attributes
    Q_to_file["graph"]["nparray_node_attributes"] = nparray_node_attributes
    Q_to_file["graph"][
        "list_of_nparrays_node_attributes"
    ] = list_of_nparrays_node_attributes
    Q_to_file["graph"]["nparray_edge_attributes"] = nparray_edge_attributes
    Q_to_file["graph"][
        "list_of_nparrays_edge_attributes"
    ] = list_of_nparrays_edge_attributes

    return Q_to_file


def decode_JSON_quiver_rep_arrays(Q_from_file):
    """
    Custom JSON decoding for NetworkX DiGraphs containing NumPy arrays.

    Parameters:
        Q_from_file: An object that can be passed to networkx.readwrite.json_graph.node_link_data

    Returns:
        The object obtained from Q_from_file by iterating through graph, node, and edge attributes generated by
            encode_JSON_quiver_rep_arrays, converting each list-of-lists to type numpy.ndarray or to a list of arrays
    """
    Q_as_dict = Q_from_file.copy()

    # Get array-valued attributes
    nparray_graph_attributes = Q_as_dict["graph"]["nparray_graph_attributes"]
    list_of_nparrays_graph_attributes = Q_as_dict["graph"][
        "list_of_nparrays_graph_attributes"
    ]
    nparray_node_attributes = Q_as_dict["graph"]["nparray_node_attributes"]
    list_of_nparrays_node_attributes = Q_as_dict["graph"][
        "list_of_nparrays_node_attributes"
    ]
    nparray_edge_attributes = Q_as_dict["graph"]["nparray_edge_attributes"]
    list_of_nparrays_edge_attributes = Q_as_dict["graph"][
        "list_of_nparrays_edge_attributes"
    ]

    # Convert each attribute to an array or list of arrays
    for key in nparray_graph_attributes:
        Q_as_dict["graph"][key] = np.array(Q_as_dict["graph"][key])
    for key in list_of_nparrays_graph_attributes:
        Q_as_dict["graph"][key] = [np.array(b) for b in Q_as_dict["graph"][key]]

    for key in nparray_node_attributes:
        for i in range(len(Q_as_dict["nodes"])):
            Q_as_dict["nodes"][i][key] = np.array(Q_as_dict["nodes"][i][key])
    for key in list_of_nparrays_node_attributes:
        for i in range(len(Q_as_dict["nodes"])):
            Q_as_dict["nodes"][i][key] = [
                np.array(b) for b in Q_as_dict["nodes"][i][key]
            ]

    for key in nparray_edge_attributes:
        for i in range(len(Q_as_dict["edges"])):
            Q_as_dict["edges"][i][key] = np.array(Q_as_dict["edges"][i][key])
    for key in list_of_nparrays_edge_attributes:
        for i in range(len(Q_as_dict["edges"])):
            Q_as_dict["edges"][i][key] = [
                np.array(b) for b in Q_as_dict["edges"][i][key]
            ]

    return Q_as_dict


class NumPyArrayEncoder(JSONEncoder):
    """
    Custom JSONEncoder class for EdgeMapRegression objects.

    Parameters:
        obj: An object arising as a dictionary value

    Returns:
        The default return of the JSONEncoder class, with NumPy arrays converted to lists
    """

    def default(self, obj):
        # Converts instances of NumPy arrays to lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class EdgeMapDecoder(JSONDecoder):
    """
    Custom JSONDecoder class for EdgeMapRegression objects.

    Parameters:
        edge_maps_from_file: A dict-of-dicts whose values are lists-of-lists

    Returns:
        The object obtained from edge_maps_from_file by converting each list-of-lists to a NumPy array
    """

    def decode(self, edge_maps_from_file):
        edge_maps = super().decode(edge_maps_from_file)
        # Decode lists-of-lists as NumPy arrays
        maps_lists_to_arrays = decode_edge_map_collection(edge_maps)
        return maps_lists_to_arrays


class NetworkXQuiverRepresentationEncoder(JSONEncoder):
    """
    Custom JSONEncoder class for NetworkX quiver representations.

    Parameters:
        Q: A NetworkX DiGraph / quiver representation

    Returns:
        The default return of JSONEncoder (to be written to file)
    """

    def default(self, Q):
        # Convert NetworkX DiGraph to JSON-compatible format
        Q_as_dict = json_graph.node_link_data(Q, edges="edges")

        # Encode NumPy arrays as lists
        Q_arrays_to_lists = encode_JSON_quiver_rep_arrays(Q_as_dict)

        return Q_arrays_to_lists


class NetworkXQuiverRepresentationDecoder(JSONDecoder):
    """
    Custom JSONDecoder class for NetworkX quiver representations.

    Parameters:
        Q_from_file: The default return of JSONDecoder.decode

    Returns:
        The decoded quiver representation as a NetworkX DiGraph
    """

    def decode(self, Q_from_file):
        Q_from_JSON = super().decode(Q_from_file)

        # Decode numpy arrays as lists
        Q_lists_to_arrays = decode_JSON_quiver_rep_arrays(Q_from_JSON)

        # Convert Python dictionary to networkx digraph
        Q = json_graph.node_link_graph(Q_lists_to_arrays, edges="edges")

        return Q
