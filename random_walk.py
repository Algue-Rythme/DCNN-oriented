import tensorflow as tf


def from_adjacency_to_transition(adj, sparse=False):
    reduce_fn = tf.sparse.reduce_sum if sparse else tf.reduce_sum
    row_sums = reduce_fn(adj, axis=1, keepdims=True)
    row_sums = tf.maximum(row_sums, tf.constant(1., dtype=tf.float32))
    transition = adj / row_sums
    return transition


def get_power_serie(p, max_step):
    """Return the concatenation of power serie of transition matrix P.

    It follows the choice of the paper for the shape of the output.

    Args:
        p: the transition matrix
        max_step: maximum number of steps of the random walk

    Returns:
        p_star: tensor of shape NxHxN with H=nb_hops if include_identity is False,
                of H=nb_hops+1 otherwise, and N=graph.nb_nodes in both cases
    """
    p_seq = [tf.eye(num_rows=p.shape[0])]
    for _ in range(max_step-1):
        p_seq.append(tf.matmul(p_seq[-1], p))
    p_star = tf.stack(p_seq, axis=1)
    return p_star


def get_naive_stationary_distribution(p):
    n = p.shape[0]
    # n**2 is the average length of a random walk in a graph before visiting
    # every node, we need at least this to detect transient states
    max_it = (n ** 2)
    p_lim = tf.math.pow(p, max_it)
    # average instead of only one column for improved robustness
    # one should check the error introduced by this method
    return tf.reduce_mean(p_lim, axis=0)


def get_reversed_time_distribution(p):
    """Return the matrix of transition corresponding to the reversed time Markov Chain.

    Args:
        p: the transition matrix

    Returns:
        p_r: the transition matrix of the reversed Markov Chain
    """
    num_nodes = int(tf.shape(p)[0])
    mu = get_naive_stationary_distribution(p)
    mu_inversed = tf.math.divide_no_nan(tf.ones(shape=mu.shape), mu)
    mu = tf.broadcast_to(mu, [num_nodes, num_nodes])
    mu_inversed = tf.broadcast_to(mu_inversed, [num_nodes, num_nodes])
    p_r = tf.transpose(p) * mu * tf.transpose(mu_inversed)
    print(tf.shape(p_r))
    return p_r
