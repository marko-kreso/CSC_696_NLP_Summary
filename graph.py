from typing import List
from typing import Type
class Graph:
    nodes:List[Type['Node']] = list()
    neighbors = dict()

    def __init__(self, weighted_adjacency_list):
        self.weighted_list = weighted_adjacency_list
        self.n = weighted_adjacency_list.shape[0]


        for i in range(self.n):
            self.nodes.append(Node(i, 1/self.n))

        for node in self.nodes:
            self.neighbors[node] = [self.nodes[j] for j in range(self.n) if self.edge_weight(node,self.nodes[j]) > 0]

        

    def get_num_nodes(self):
        return self.n
        

    def get_nodes(self):
        return self.nodes


    def edge_weight(self, node1,node2):
        idx1 = self.nodes.index(node1)
        idx2 = self.nodes.index(node2)

        return self.weighted_list[idx1,idx2]

    def get_neighbors(self, node):
        return self.neighbors[node]

    def get_scores(self):
        return [node.score for node in self.nodes] 

class Node:
    def __init__(self, index, score):
        self.index = index
        self.score = score
    def set_score(self, new_score):
        self.score = new_score 
    
    def get_index(self):
        return self.get_index

    def get_score(self):
        return self.score