import json
from json import JSONEncoder, JSONDecoder

import pandas as pd
import numpy as np

import networkx as nx
from networkx.readwrite import json_graph


# JSON decoding, for a dict-of-dicts containing a numpy array for each pair of nodes
# For encoding, just use NumpyArrayEncoder
def decode_edge_map_collection(edge_maps_from_file):
    maps_as_dict = edge_maps_from_file.copy()
    node_list = maps_as_dict.keys()
    for head in node_list:
        tails = maps_as_dict[head].keys()
        for tail in tails:
            maps_as_dict[head][tail] = np.array(maps_as_dict[head][tail])
    return maps_as_dict


# JSON networkx graph encoding, to be compatible with the output of networkx.readwrite.json_graph.node_link_data
def encode_JSON_quiver_rep_arrays(Q_as_dict):
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

    # Save these attributes for the decoder
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

    return Q_to_file


# JSON networkx graph decoding, to restore nparrays encoded as lists
def decode_JSON_quiver_rep_arrays(Q_from_file):
    Q_as_dict = Q_from_file.copy()

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


# JSON encoder/decoder subclasses currently used to encode a dict-of-dicts of numpy arrays, for all possible edges
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class EdgeMatrixDecoder(JSONDecoder):
    def decode(self, edge_maps_from_file):
        edge_maps = super().decode(edge_maps_from_file)
        # Decode numpy arrays as lists
        maps_lists_to_arrays = decode_edge_map_collection(edge_maps)
        return maps_lists_to_arrays


# JSON encoder/decoder subclasses for quiver representations


class NetworkXQuiverRepresentationEncoder(JSONEncoder):
    def default(self, Q):
        # Convert networkx digraph to Python dictionary
        Q_as_dict = json_graph.node_link_data(Q, edges="edges")

        # Encode numpy arrays as lists
        Q_arrays_to_lists = encode_JSON_quiver_rep_arrays(Q_as_dict)

        return Q_arrays_to_lists


class NetworkXQuiverRepresentationDecoder(JSONDecoder):
    def decode(self, Q_from_file):
        Q_from_JSON = super().decode(Q_from_file)

        # Decode numpy arrays as lists
        Q_lists_to_arrays = decode_JSON_quiver_rep_arrays(Q_from_JSON)

        # Convert Python dictionary to networkx digraph
        Q = json_graph.node_link_graph(Q_lists_to_arrays, edges="edges")

        return Q
