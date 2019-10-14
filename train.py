import argparse
import math
import tensorflow as tf

import dataset
import dcnn
import labeled_graph
import utils


def learn_toy(
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


def node_classification(dataset_reader, orientation, forward, reverse,
                        num_hops, num_epochs, batch_size, split_ratio,
                        bias=False, optimizer='adam'):
    with tf.device('/cpu:0'):
        p, features, labels, num_classes = dataset_reader(forward, reverse)
        features = dcnn.sparse_diffuse_features(orientation, p, features, num_hops)
        del p  # save memory
        train_set, test_set = utils.shuffle_and_split(features, labels, split_ratio)
        num_train_examples = int(tf.shape(train_set[0])[0])
        dcnn_num_hops = int(tf.shape(train_set[0])[1])
        num_features = int(tf.shape(train_set[0])[2])
        train_set = tf.data.Dataset.from_tensor_slices(train_set).repeat()
        train_set = train_set.shuffle(batch_size * 8).batch(batch_size)
        test_set = tf.data.Dataset.from_tensor_slices(test_set).batch(256)
        model = dcnn.DCNN(num_features, dcnn_num_hops, num_classes, bias)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    with tf.device('/gpu:0'):
        num_batchs = int(math.ceil(num_train_examples / batch_size))
        for epoch in range(num_epochs):
            print('Epoch %d'%(epoch+1))
            model.fit(train_set, epochs=1, steps_per_epoch=num_batchs)
            model.evaluate(test_set)
            print('')


if __name__ == "__main__":
    # tf.random.set_seed(146)
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. cora, pubmed-diabetes or toy (sanity_check)')
    args = parser.parse_args()
    if args.task == 'cora':
        node_classification(dataset.read_cora,
                            orientation='forward',
                            forward=True,
                            reverse=True,
                            num_hops=3,
                            num_epochs=50,
                            batch_size=32,
                            split_ratio=0.1,
                            bias=False)
    elif args.task == 'pubmed-diabetes':
        node_classification(dataset.read_pubmed_diabetes,
                            orientation='forward',
                            forward=True,
                            reverse=True,
                            num_hops=3,
                            num_epochs=50,
                            batch_size=32,
                            split_ratio=0.1,
                            bias=True)
    elif args.task == 'toy':
        learn_toy('tests/petersen_topology_10.txt',
                  'tests/toy_features_10.txt',
                  'forward', num_hops=4)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
