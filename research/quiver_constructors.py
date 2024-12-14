import json

import pandas as pd
import numpy as np

import networkx as nx

from linear_algebra_utils import *

from scipy.stats import gaussian_kde


class DataQuiverConstructor:
    def __init__(self, data, node_columns, edge_maps, ref_index = 0, node_limit = None, build_direction='incoming-outgoing'):
        print(f"Generating quiver Q{ref_index:02}!")
        self.data = data
        self.node_columns = node_columns
        self.all_edge_maps = edge_maps
        self.ref_index = ref_index

        self.nodes = list(edge_maps.keys())
        self.n_pts = len(data)

        if build_direction == 'random':
            self.build_direction = np.random.choice(['incoming', 'outgoing', 'incoming-outgoing'])
        else:
            self.build_direction = build_direction

        if node_limit:
            self.node_limit = max([ min([node_limit, len(self.nodes)]), 1])
        else:
            self.node_limit = len(self.nodes)


        # Store data for the ref node (first node added to the quiver)
        self.ref_node = self.nodes[self.ref_index]
        self.ref_node_data = self.data[self.node_columns[self.ref_node]]

        self.quiver = None

        # Temporary values when recursively adding a new edge
        self.unused_nodes = None
        self.last_added_node = None
        self.new_edges = None

        # Store running list of edges to include in quiver
        self.quiver_edges = []
        
        # Store the relevant info when adding a node to the quiver
        self.quiver_node_info = {
            node: {
                'direction': None,
                'layer': np.inf,
                'entropy': np.inf,
                'path map': None
            } 
        for node in self.nodes}

        # Store nodes according to layer
        self.quiver_nodes_by_layer = {
            d: [] 
            for d in range(-len(self.nodes),len(self.nodes) + 1)
        }
        
        # Store a running list of possible next edges with relevant info
        self.next_edge_possibilities = []


        
        # Record info for the start node
        self.quiver_nodes_by_layer[0].append(self.ref_node)
        self.quiver_node_info[self.ref_node].update({
            'direction': 'ref',
            'layer': 0,
            'entropy': 0.0,
            'path map': np.eye(len(self.node_columns[self.ref_node]))
        })

        self.last_added_node = str(self.ref_node)
        self.unused_nodes = [node for node in self.nodes if node != self.ref_node]

    
        # Select edges for the quiver, then construct the quiver
        for i in range(self.node_limit - 1):
            self.fit_next_edge()
        self.create_quiver_representation()


    

    ########## Auxiliary math and retrieval functions
    
    def negative_log_pdf(self, x, pdf):
        pdfx = pdf.evaluate(x)
        if np.abs(pdfx) < 1e-20:
            return 0
        else:
            return -np.log(pdfx)[0]

    def entropy_score(self, new_edge):
        residuals = new_edge['residuals']
        path_tail = new_edge['path tail']
        path_head = new_edge['path head']

        pdf = gaussian_kde(residuals.T)
        sigma = pdf.factor

        # Scaling to correct for dimensions of domain/codomain of path maps
        d1 = len(self.node_columns[path_tail])
        d2 = len(self.node_columns[path_head])
        # This is an experimental guess
        scaling_factor = 1/(d1*d2)
        
        num_samples = 100
        samples = []
        for i in range(num_samples):
            j = np.random.randint(self.n_pts)
            pj = residuals.iloc[j].to_numpy()
            xj = np.random.multivariate_normal(pj, sigma*np.eye(len(pj)))
            samples.append(self.negative_log_pdf(xj,pdf))
        return scaling_factor*np.array(samples).mean()

    def compose_path_map(self, new_edge):
        direction = new_edge['direction']
        tail = new_edge['tail']
        head = new_edge['head']
        if direction == 'incoming':
            closer_node = head
            farther_node = tail
        elif direction == 'outgoing':
            closer_node = tail
            farther_node = head
        closer_node_path_map = self.quiver_node_info[closer_node]['path map']
        new_edge_map = self.all_edge_maps[tail][head]
        if direction == 'incoming':
            path_map = closer_node_path_map.dot(new_edge_map)  
        elif direction == 'outgoing':
            path_map = new_edge_map.dot(closer_node_path_map)
        return path_map

    def has_vanishing_path_map(self, new_edge):
        path_map = new_edge['path map']
        return np.abs(path_map).sum() < 0.01

    def node_data_actual(self, new_edge):
        direction = new_edge['direction']
        if direction == 'incoming':
            return self.ref_node_data.copy()
        elif direction == 'outgoing':
            head = new_edge['head']
            return self.data[self.node_columns[head]]
            
    def node_data_predicted(self, new_edge):
        direction = new_edge['direction']
        path_map = new_edge['path map']
        if direction == 'incoming':
            tail = new_edge['tail']
            data_to_map = self.data[self.node_columns[tail]]
        elif direction == 'outgoing':
            data_to_map = self.ref_node_data.copy() 
        predicted = data_to_map.dot(path_map.T) 
        if direction == 'incoming':
            predicted.columns = self.node_columns[self.ref_node]
        elif direction == 'outgoing':
            predicted.columns = self.node_columns[new_edge['head']]
        return predicted

    def enumerate_edges(self, direction):
        if direction == 'incoming':
            return [(node, self.last_added_node) for node in self.unused_nodes] 
        elif direction == 'outgoing':
            return [(self.last_added_node, node) for node in self.unused_nodes]  

    def path_tail(self, edge, direction):
        if direction == 'incoming':
            return edge[0]
        elif direction == 'outgoing':
            return self.ref_node

    def path_head(self, edge, direction):
        if direction == 'incoming':
            return self.ref_node
        elif direction == 'outgoing':
            return edge[1]
    
    def construct_edge_objects(self, edges, direction):
        return [
            {
                'direction': direction,
                'edge': edge,
                'tail': edge[0],
                'head': edge[1],
                'path tail': self.path_tail(edge, direction),
                'path head': self.path_head(edge, direction),
                'edge map': self.all_edge_maps[edge[0]][edge[1]],
            }
        for edge in edges
        ]   

    def layer_of_new_node(self, new_edge):
        direction = new_edge['direction']
        if direction == 'incoming':
            closer_node = new_edge['head']
            return self.quiver_node_info[closer_node]['layer'] - 1
        elif direction == 'outgoing':
            closer_node = new_edge['tail']
            return self.quiver_node_info[closer_node]['layer'] + 1

    def new_node_from_new_edge(self, new_edge):
        direction = new_edge['direction']
        if direction == 'incoming':
            return new_edge['tail']
        elif direction == 'outgoing':
            return new_edge['head']

    def previous_node_from_new_edge(self, new_edge):
        direction = new_edge['direction']
        if direction == 'incoming':
            return new_edge['head']
        elif direction == 'outgoing':
            return new_edge['tail']




    ########## Choose a new edge for the quiver by first adding eligible edges incident to the last-added node     

    def get_new_edges(self):
        # On a case by case basis, construct the edge objects in either direction
        direction = self.quiver_node_info[self.last_added_node]['direction']
        if (direction == 'ref' and self.build_direction == 'incoming-outgoing'):
            edges_incoming = self.enumerate_edges('incoming')
            edges_outgoing = self.enumerate_edges('outgoing')
            self.new_edges = self.construct_edge_objects(edges_incoming, 'incoming') + self.construct_edge_objects(edges_outgoing, 'outgoing') 
        else:
            if direction == 'ref':
                direction = self.build_direction
            edges = self.enumerate_edges(direction)   
            self.new_edges = self.construct_edge_objects(edges, direction)

        # Record path maps (composition of edge maps between the reference node up to and including the current edge)
        for new_edge in self.new_edges:  
            new_edge['path map'] = self.compose_path_map(new_edge)

        # Delete edges whose path map (approximately) vanishes
        self.new_edges = [new_edge for new_edge in self.new_edges if not self.has_vanishing_path_map(new_edge)]

        # Now record residuals (base data against path-mapped data) and their entropy scores
        for new_edge in self.new_edges:
            data_actual = self.node_data_actual(new_edge)
            data_predicted = self.node_data_predicted(new_edge)
            new_edge['residuals'] = data_actual - data_predicted      
            
            new_edge['entropy'] = self.entropy_score(new_edge)

        self.next_edge_possibilities.extend(self.new_edges)


    def add_best_edge(self):
        entropy_scores = [e['entropy'] for e in self.next_edge_possibilities]
        new_edge_index = entropy_scores.index(min(entropy_scores))
        new_edge = self.next_edge_possibilities[new_edge_index]
        previous_node = self.previous_node_from_new_edge(new_edge)
        new_node = self.new_node_from_new_edge(new_edge)

        new_layer = self.layer_of_new_node(new_edge)
        new_direction = new_edge['direction']
        new_entropy = new_edge['entropy']
        new_path_map = new_edge['path map']
        

        # Add new edge and record info to references
        self.quiver_edges.append(new_edge)
        self.quiver_nodes_by_layer[new_layer].append(new_node)
        self.quiver_node_info[new_node].update({
            'direction': new_direction,
            'layer': new_layer,
            'entropy': new_entropy,
            'path map': new_path_map
        })    


        self.last_added_node = str(new_node)

        
        print(f"{new_edge['edge']} to layer {new_layer}")
    
    def remove_unusable_edges(self):
        new_node = self.last_added_node
        self.unused_nodes = [node for node in self.unused_nodes if node != new_node]
        self.next_edge_possibilities = [e for e in self.next_edge_possibilities if (e['head'] != new_node and e['tail'] != new_node)]
        



    

    ########## Recursive step to add new edges
    
    def fit_next_edge(self):
        self.get_new_edges()
        self.add_best_edge()
        self.remove_unusable_edges()



        

    ########## Construct quiver representation from complete list of chosen edges
    
    def create_quiver_representation(self):
        self.quiver = nx.DiGraph()
        self.quiver.graph.update({'name': f"Q{self.ref_index:02}"})
        # Add edges and edge attribute (representation maps)
        for edge in self.quiver_edges:
            tail = edge['tail']
            head = edge['head']
            self.quiver.add_edge(tail,head)
            self.quiver.edges[tail,head].update({
                'rep map': edge['edge map']
            })

        # Add node attributes for dimension and basis
        nodes = list(self.quiver.nodes)
        dimensions = {node: len(self.node_columns[node]) for node in nodes}
        bases = construct_standard_bases(dimensions)
        nx.set_node_attributes(self.quiver,dimensions,'dimension')
        nx.set_node_attributes(self.quiver,bases,'basis')


















            
