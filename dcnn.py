import tensorflow as tf
import random_walk


def diffuse_features(orientation, p, features, num_hops):
    """Returns the product of diffusion operator with the features.

    Args:
        orientation: whether to walk in reverse time
        p: transition matrix (float32 tensor)
        features: float32 tensor of shape NxF
        num_hops: integer, number of steps in random walk
        include_identity: include 0-length steps

    Returns:
        float32 tensor of shape NxHxF
    """
    p_star = random_walk.get_power_serie(p, num_hops)
    if orientation == 'reversed':
        p_r = random_walk.get_reversed_time_distribution(p)
        p_r_star = random_walk.get_power_serie(p_r, num_hops)
        diffusion = tf.concat([p_star, p_r_star], axis=1)
    else:
        diffusion = p_star
    y = tf.einsum("ijl,lk->ijk", diffusion, features)
    return y


class DiffusionConvolution(tf.keras.layers.Layer):

    def __init__(self, num_features, num_hops, bias=True):
        super(DiffusionConvolution, self).__init__()
        self._dim_features = num_features
        self._num_hops = num_hops
        self._kernel = None
        self._bias = None
        self._has_bias = bias

    def build(self, unused_input_shape):
        self._kernel = self.add_weight('kernel', shape=[self._num_hops, self._dim_features])
        if self._has_bias:
            self._bias = self.add_weight('bias', shape=[self._num_hops, self._dim_features])

    def call(self, x):
        """Apply diffusion convolution on input graph.

        Node dimension corresponds to batch size.

        Args:
            x: float32 Tensor of shape NxHxF

        Returns:
            y: float32 Tensor of shape NxHxF
        """
        y = tf.einsum('jk,ijk->ijk', self._kernel, x)
        if self._has_bias:
            y = y + self._bias
        return y


class DCNN(tf.keras.Model):

    def __init__(self, num_features, num_hops, output_dim, bias=True):
        super(DCNN, self).__init__(name='')
        self._num_features = num_features
        self._num_hops = num_hops
        self._output_dim = output_dim
        self._dc = DiffusionConvolution(self._num_features, self._num_hops, bias)
        self._fc = tf.keras.layers.Dense(output_dim)


    def call(self, x):
        """Return the logits/regression for each node.

        Args:
            x: float32 Tensor of shape NxHxF

        Returns:
            y: float32 Tensor of shape Nx1
        """
        y = self._dc(x)
        y = tf.nn.relu(y)
        y = tf.reshape(y, shape=[-1, self._num_features * self._num_hops])
        y = self._fc(y)
        return y
