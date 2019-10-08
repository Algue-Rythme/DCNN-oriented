import numpy as np
import tensorflow as tf


class DenseLabelizedGraph:

    def __init__(self):
        self._num_nodes = 0
        self._num_edges = 0
        self._features = None
        self._num_features = 0
        self._adj = None
        self._transition = None

    # initialization methods

    def load_from_raw_matrices(self, num_nodes, num_edges, features, adj):
        self._num_nodes = num_nodes
        self._num_edges = num_edges
        self._features = features
        self._adj = adj

    def load_from_text(self, filename):
        with open(filename, "r") as file:
            num_nodes = int(input(file))
            self._nb_nodes = num_nodes
            features = []
            for _ in range(self.num_nodes):
                node_features = file.readline().split()
                features.append([float(feature) for feature in node_features])
            self._features = np.array(features, dtype=np.float32)
            adj = []
            for _ in range(self.num_nodes):
                adj_line = file.readline()
                adj.append([float(digit) for digit in adj_line])
            self._adj = np.array(adj, dtype=np.float32)
            self._num_edges = np.count_nonzero(self._adj)

    # getter methods

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def features(self):
        return self._features

    @property
    def adj(self):
        return self._adj

    @property
    def num_features(self):
        if self._features is None:
            return 0
        return self._features.shape[1]

    @property
    def transition_matrix(self):
        if self._transition is None:
            row_sums = tf.reduce_sum(self.adj, axis=1, keepdims=True)
            self._transition = self.adj / row_sums
        return self._transition

    def get_transpose_graph(self):
        graph = DenseLabelizedGraph()
        graph.load_from_raw_matrices(
            self.num_nodes, self.num_edges,
            self.features, self.adj.T)
        return graph

    # printing and vizualization methods

    def summary(self):
        print("Graph with %d nodes and %d edges"%(self.num_nodes, self.num_edges))
