from typing import List
from typing import Type
import numpy as np
class Graph:

    def __init__(self, weighted_adjacency_list):
        self.nodes:List[Type['Node']] = list()
        self.neighbors = dict()
        self.neighbor_weights = np.empty()

        self.weighted_list = weighted_adjacency_list
        self.n = weighted_adjacency_list.shape[0]
        self.scores = np.full((self.n,1),1/self.n,)


        for i in range(self.scores.shape[0]):
            self.nodes.append(Node(i, 1/self.n))

        for node in self.nodes:
            self.neighbors[node] = [self.nodes[j] for j in range(self.n) if self.edge_weight(node,self.nodes[j]) > 0]
        

    def get_num_nodes(self):
        return self.n
        
    def calculate_weights(self):
        #Calculate sum of neighbors
        self.neighbor_weights = self.weighted_list @ np.ones(self.weighted_list.shape[0])
        self.neighbor_weights = (self.weighted_list.T @ np.inverse(np.diag(self.weighted_list))) *  self.scores 
        



    def get_nodes(self):
        return self.nodes


    def edge_weight(self, node1,node2):
        idx1 = self.nodes.index(node1)
        idx2 = self.nodes.index(node2)

        return self.weighted_list[idx1,idx2]

    def get_neighbors(self, node):
        return self.neighbors[node]

    def get_scores(self):
        return self.scores

class Node:
    def __init__(self, index, score):
        self.index = index
        self.score = score

    def set_score(self, new_score):
        self.score = new_score 
        self.scores[self.index] = new_score
    
    def get_index(self):
        return self.get_index

    def get_score(self):
        return self.score