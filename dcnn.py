import tensorflow as tf

import random_walk


def map_symmetric_features(graph, num_hops):
    p_star, p_r_star = random_walk.get_diffusion_operators(graph, num_hops)
    pp = tf.concat([p_star, p_r_star], axis=1)
    xx = tf.concat([graph.features, graph.features], axis=1)
    yy = tf.einsum("ilj,lk->ijk", pp, xx)
    return yy


def map_forward_features(graph, num_hops):
    p = graph.transition_matrix
    p_star = random_walk.get_power_serie(p, num_hops)
    x = graph.features
    y = tf.einsum("ilj,lk->ijk", p_star, x)
    return y


class DiffusionConvolution(tf.keras.layers.Layer):

    def __init__(self, nb_features, nb_hops):
        super(DiffusionConvolution, self).__init__()
        self._dim_features = nb_features
        self._num_hops = num_hops
        self._kernel = None

    def build(self, unused_input_shape):
        self._kernel = self.add_variable('kernel',
                                         shape=[self._num_hops, self._dim_features])

    def call(self, x):
        y = tf.einsum('jk,ijk->ijk', self._kernel, x)
        return y


class DCNN(tf.keras.Model):

    def __init__(self, num_features, num_hops, num_classes):
        super(DCNN, self).__init__(name='')
        self._dc = DiffusionConvolution(num_features, num_hops)
        self._fc1 = tf.keras.layers.Dense(256)
        self._fc2 = tf.keras.layers.Dense(num_classes)


    def call(self, x):
        x = self._dc(x)
        x = tf.nn.relu(x)
        x = self._fc1(x)
        x = tf.nn.relu(x)
        x = self._fc2(x)
        x = tf.nn.softmax(x)
        return x
