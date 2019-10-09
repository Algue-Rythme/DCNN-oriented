import tensorflow as tf

import dcnn
import labeled_graph
import utils


def train(model, x_train, y_train, loss_fn, nb_epochs):
    optimizer = tf.optimizers.Adam()
    for epoch in range(nb_epochs):
        with tf.GradientTape() as tape:
            output = model(x_train)
            loss_value = loss_fn(y_train, output)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        print('epoch %d      loss value: %f'%(epoch, float(loss_value.numpy().mean())))
    print('Accuracy test:')
    print(tf.nn.softmax(output))
    print(y_train)


def node_classification_task(
        topology_file, features_file,
        orientation, num_hops, target_dims=None,
        include_identity=True):
    if target_dims is None:
        target_dims = [-1]
    graph = labeled_graph.DenseLabeledGraph()
    graph.load_topology_from_text(topology_file)
    graph.load_features_from_text(features_file)
    graph.summary(verbose=1)
    x = utils.extract_inputs(graph.features, target_dims)
    y = utils.extract_targets(graph.features, target_dims)
    num_features = tf.shape(x)[1]
    p = graph.transition_matrix
    y, num_classes = utils.to_categorical(tf.squeeze(y, axis=1))
    if orientation == 'reversed':
        num_features *= 2
    x = dcnn.diffuse_features(orientation, p, x, num_hops, include_identity)
    if include_identity:
        num_hops += 1
    model = dcnn.DCNN(num_features, num_hops, num_classes, bias=True)
    loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
    train(model, x, y, loss_fn, nb_epochs=2*1000)


def learn_cora_dataset():
    pass


if __name__ == "__main__":
    tf.random.set_seed(146)
    node_classification_task('tests/petersen_topology_10.txt',
                             'tests/toy_features_10.txt',
                             'forward', num_hops=3)
