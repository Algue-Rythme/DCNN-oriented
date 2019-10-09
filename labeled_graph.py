import tensorflow as tf


class DenseLabeledGraph:

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

    def load_topology_from_text(self, filename):
        with open(filename, "r") as file:
            num_nodes = int(file.readline())
            self._num_nodes = num_nodes
            adj = []
            for _ in range(self.num_nodes):
                adj_line = file.readline()  # remove \n final character
                adj.append([float(digit) for digit in adj_line[:-1]])
            self._adj = tf.constant(adj, dtype=tf.float32)
            self._num_edges = tf.math.count_nonzero(self._adj)

    def load_features_from_text(self, filename):
        with open(filename, "r") as file:
            features = []
            for _ in range(self.num_nodes):
                node_features = file.readline().split()
                features.append([float(feature) for feature in node_features])
            self._features = tf.constant(features, dtype=tf.float32)

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
            row_sums = tf.maximum(row_sums, tf.constant(1., dtype=tf.float32))
            self._transition = self.adj / row_sums
        return self._transition

    def get_transpose_graph(self):
        graph = DenseLabeledGraph()
        graph.load_from_raw_matrices(
            self.num_nodes, self.num_edges,
            self.features, tf.transpose(self.adj))
        return graph

    # printing and vizualization methods

    def summary(self, verbose=0):
        print("Graph with %d nodes and %d edges"%(self.num_nodes, self.num_edges))
        if verbose == 1:
            print('Features vector: ', self.features)
            print('Adjacency matrix: ', self._adj)
