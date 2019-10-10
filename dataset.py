import tensorflow as tf
import numpy as np


def read_cora(forward, reverse):
    class_names = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
		                 'Probabilistic_Methods', 'Reinforcement_Learning',
		                 'Rule_Learning', 'Theory']
    class_indexes = dict(zip(class_names,range(len(class_names))))
    indexes = dict()
    features = []
    labels = []
    print('opening cora/cora.content...')
    nb_papers = 2708
    with open('cora/cora.content') as f:
        tokens = f.readline().split()
        progbar = tf.keras.utils.Progbar(nb_papers)
        while tokens:
            paper_id = int(tokens[0])
            indexes[paper_id] = len(indexes)
            node_features = np.array([float(f) for f in tokens[1:-1]], dtype=np.float32)
            features.append(node_features)
            labels.append(int(class_indexes[tokens[-1]]))
            tokens = f.readline().split()
            progbar.update(len(indexes))
    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.int64)
    print('opening cora/cora.cites...')
    adj = np.zeros(shape=(len(indexes),len(indexes)), dtype=np.float32)
    with open('cora/cora.cites') as f:
        tokens = f.readline().split()
        max_num_eges = 5429
        progbar = tf.keras.utils.Progbar(max_num_eges)
        num_edges = 0
        while tokens:
            cited = indexes[int(tokens[0])]
            citing = indexes[int(tokens[1])]
            if forward:
                adj[citing, cited] = 1.
            if reverse:
                adj[cited, citing] = 1.
            tokens = f.readline().split()
            num_edges += 1
            progbar.update(num_edges)
    return adj, features, labels
