import argparse
import math
import tensorflow as tf

import dataset
import dcnn
import labeled_graph
import utils
import random_walk


def train(model, training_set, loss_fn, num_epochs, num_batchs):
    optimizer = tf.optimizers.Adam()
    for _ in range(num_epochs):
        progbar = tf.keras.utils.Progbar(num_batchs)
        for batch, (x_train, y_train) in enumerate(training_set.take(num_batchs)):
            # print(batch)
            with tf.GradientTape() as tape:
                output = model(x_train)
                loss_value = loss_fn(y_train, output)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            progbar.update(batch+1, [('loss', float(loss_value.numpy().mean()))])


def node_classification_task(
        topology_file, features_file,
        orientation, num_hops, target_dims=None):
    if target_dims is None:
        target_dims = [-1]
    graph = labeled_graph.DenseLabeledGraph()
    graph.load_topology_from_text(topology_file)
    graph.load_features_from_text(features_file)
    graph.summary(verbose=1)
    x = utils.extract_inputs(graph.features, target_dims)
    y = utils.extract_targets(graph.features, target_dims)
    num_nodes = int(tf.shape(x)[0])
    num_features = tf.shape(x)[1]
    p = graph.transition_matrix
    y, num_classes = utils.to_categorical(tf.squeeze(y, axis=1))
    if orientation == 'reversed':
        num_hops *= 2
    x = dcnn.diffuse_features(orientation, p, x, num_hops)
    toy_dataset = tf.data.Dataset.from_tensor_slices((x, y)).repeat().batch(16)
    model = dcnn.DCNN(num_features, num_hops, num_classes, bias=True)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(toy_dataset, epochs=10, steps_per_epoch=768)
    model.evaluate(toy_dataset, steps=num_nodes)


def learn_cora_dataset(orientation, num_hops, num_epochs, batch_size, split_ratio):
    adj, features, labels = dataset.read_cora(forward=True, reverse=True)
    p = random_walk.from_adjacency_to_transition(tf.constant(adj))
    features = dcnn.diffuse_features(orientation, p, features, num_hops)
    train_set, test_set = utils.shuffle_and_split(features, labels, split_ratio)
    num_train_examples = int(tf.shape(train_set[0])[0])
    train_set = tf.data.Dataset.from_tensor_slices(train_set).repeat()
    train_set = train_set.shuffle(batch_size * 8).batch(batch_size)
    num_features, num_classes = 1433, 7
    if orientation == 'reversed':
        num_hops *= 2
    model = dcnn.DCNN(num_features, num_hops, num_classes, bias=False)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    test_set = tf.data.Dataset.from_tensor_slices(test_set).batch(256)
    num_batchs = int(math.ceil(num_train_examples / batch_size))
    for epoch in range(num_epochs):
        print('Epoch %d'%(epoch+1))
        model.fit(train_set, epochs=1, steps_per_epoch=num_batchs)
        model.evaluate(test_set)
        print('')


if __name__ == "__main__":
    # tf.random.set_seed(146)
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. cora for Cora dataset, and toy for sanity check purposes.')
    args = parser.parse_args()
    if args.task == 'cora':
        learn_cora_dataset('forward', num_hops=3, num_epochs=50, batch_size=32, split_ratio=0.66)
    elif args.task == 'toy':
        node_classification_task('tests/petersen_topology_10.txt',
                                 'tests/toy_features_10.txt',
                                 'forward', num_hops=4)
    else:
        print('Unknown task %s'%args.task)
