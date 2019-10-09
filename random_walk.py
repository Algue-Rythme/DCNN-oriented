import tensorflow as tf


def get_power_serie(p, num_hops, include_identity=True):
    """Return the concatenation of power serie of transition matrix P.

    It follows the choice of the paper for the shape of the output.

    Args:
        p: the transition matrix
        num_hops: number of steps of the random walk
        include_identity: whether to include the 0-th step

    Returns:
        p_star: tensor of shape NxHxN with H=nb_hops if include_identity is False,
                of H=nb_hops+1 otherwise, and N=graph.nb_nodes in both cases
    """
    p_seq = [tf.eye(num_rows=p.shape[0])] if include_identity else [p]
    for _ in range(num_hops):
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
    mu = get_naive_stationary_distribution(p)
    # warning with transient states
    mu_inversed = tf.math.divide_no_nan(tf.ones(shape=mu.shape), mu)
    p_t = tf.transpose(p)
    p_r = tf.dot(mu_inversed, tf.dot(p_t, mu))
    return p_r
