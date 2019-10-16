import tensorflow as tf
import numpy as np

import random_walk


def read_cora_citeseer_like(forward, reverse, node_file, edge_file, num_papers, max_num_edges, class_names):
    class_indexes = dict(zip(class_names,range(len(class_names))))
    indexes = dict()
    features = []
    labels = []
    print('opening %s...'%node_file)
    with open(node_file) as f:
        tokens = f.readline().split()
        progbar = tf.keras.utils.Progbar(num_papers)
        while tokens:
            paper_id = tokens[0]
            indexes[paper_id] = len(indexes)
            node_features = np.array([float(f) for f in tokens[1:-1]], dtype=np.float32)
            features.append(node_features)
            labels.append(int(class_indexes[tokens[-1]]))
            tokens = f.readline().split()
            progbar.update(len(indexes))
    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.int64)
    print('opening %s...'%edge_file)
    with open(edge_file) as f:
        tokens = f.readline().split()
        edge_indices = []
        progbar = tf.keras.utils.Progbar(max_num_edges)
        num_edges = 0
        while tokens:
            cited = indexes[tokens[0]]
            citing = indexes[tokens[1]]
            if forward:
                edge_indices.append([citing, cited])
            if reverse:
                edge_indices.append([cited, citing])
            tokens = f.readline().split()
            num_edges += 1
            progbar.update(num_edges)
    edge_indices.sort()
    ones = tf.ones(shape=(len(edge_indices),), dtype=tf.float32)
    p = tf.sparse.SparseTensor(edge_indices, ones, dense_shape=(num_papers, num_papers))
    p = random_walk.from_adjacency_to_transition(p, sparse=True)
    num_labels = len(class_names)
    return p, features, labels, num_labels


def read_cora(forward, reverse):
    class_names = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
		                 'Probabilistic_Methods', 'Reinforcement_Learning',
		                 'Rule_Learning', 'Theory']
    num_papers = 2708
    max_num_edges = 5429
    node_file = 'cora/cora.content'
    edge_file = 'cora/cora.cites'
    return read_cora_citeseer_like(forward, reverse, node_file, edge_file, num_papers, max_num_edges, class_names)


def read_citeseer(forward, reverse):
    class_names = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']
    num_papers = 3312
    max_num_edges = 4732
    node_file = 'citeseer/citeseer.content'
    edge_file = 'citeseer/citeseer.cites'
    return read_cora_citeseer_like(forward, reverse, node_file, edge_file, num_papers, max_num_edges, class_names)


def read_pubmed_diabetes(forward, reverse):
    print('opening pubmed/NODE.paper.tab')
    with open('pubmed/NODE.paper.tab') as f:
        tokens = f.readline().split()  # remove header
        tokens = f.readline().split()  # parse attribute names
        features_ids = dict()
        for token in tokens[1:-1]:
            _, name, _ = tuple(token.split(':'))
            features_ids[name] = len(features_ids)
        num_papers = 19717
        num_features = 500
        progbar = tf.keras.utils.Progbar(num_papers)
        indexes = dict()
        features = np.zeros(shape=(num_papers, num_features), dtype=np.float32)
        labels = []
        tokens = f.readline().split()
        while tokens:
            paper_id = int(tokens[0])
            indexes[paper_id] = len(indexes)
            labels.append(float(tokens[1].split('=')[1]) - 1.)
            for token in tokens[2:-1]:
                tag, value = tuple(token.split('='))
                features_id = features_ids[tag]
                features[indexes[paper_id], features_id] = float(value)
            progbar.update(len(indexes))
            tokens = f.readline().split()
        features = tf.constant(features, dtype=tf.float32)
    print('opening pubmed/DIRECTED.cites.tab')
    with open('pubmed/DIRECTED.cites.tab') as f:
        max_num_edges = 44338
        tokens = f.readline().split()  # remove header
        tokens = f.readline().split()  # parse NO-FEATURES
        edge_indices = []
        progbar = tf.keras.utils.Progbar(max_num_edges)
        cur_edge = 0
        tokens = f.readline().split()
        while tokens:
            paper_a = indexes[int(tokens[1].split(':')[1])]
            paper_b = indexes[int(tokens[3].split(':')[1])]
            if forward:
                edge_indices.append([paper_a, paper_b])
            if reverse:
                edge_indices.append([paper_b, paper_a])
            tokens = f.readline().split()
            cur_edge += 1
            progbar.update(cur_edge)
    edge_indices.sort()
    ones = tf.ones(shape=(len(edge_indices),), dtype=tf.float32)
    p = tf.sparse.SparseTensor(edge_indices, ones, dense_shape=(num_papers, num_papers))
    p = random_walk.from_adjacency_to_transition(p, sparse=True)
    num_labels = 3
    return p, features, labels, num_labels
