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
    p_seq = [tf.eye(N=p.shape[0])] if include_identity else [p]
    for _ in range(num_hops):
        p_seq.append(tf.expand_dims(tf.matmul(p_seq[-1], p), axis=1))
    p_star = tf.concat(p_seq, axis=1)
    return p_star


def get_naive_stationary_distribution(p):
    n = p.shape[0]
    # n**2 is the average length of a random walk in a graph before visiting
    # every node, we need at least this to detect transient states
    # n**3 is a way to reach this value
    max_it = (n ** 3)
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
    p_r = tf.dot(mu_inversed, tf.dot(p.T, mu))
    return p_r


def get_diffusion_operators(graph, num_hops, include_identity=True):
    p = graph.transition_matrix
    p_r = get_reversed_time_distribution(p)
    p_star = get_power_serie(p, num_hops, include_identity)
    p_r_star = get_power_serie(p_r, num_hops, include_identity)
    return p_star, p_r_star
