import tensorflow as tf

import labelized_graph
import dcnn


def train(model, x, y, loss):
    pass


def node_classification_task(topology_file, features_file, orientation, num_hops):
    graph = labelized_graph.DenseLabelizedGraph()
    graph.load_topology_from_text(topology_file)
    graph.load_features_from_text(features_file)
    graph.summary()
    # TODO: extract useful information
    output_dim = None
    num_features = graph.num_features
    if orientation == 'reversed':
        num_features *= 2
    x, y, loss = None, None, None
    model = dcnn.DCNN(num_features, num_hops, output_dim)
    train(model, x, y, loss)


if __name__ == "__main__":
    tf.enable_v2_behavior()
    node_classification_task('test/toy_topology_10.txt',
                             'tests/toy_features_10.txt',
                             'forward', num_hops=2)
