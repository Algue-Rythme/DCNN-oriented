import tensorflow as tf


def reindex(indexes, num):
    return [(num+index)%num for index in indexes]

def extract_columns(matrix, columns):
    columns = [[col] for col in reindex(columns, tf.shape(matrix)[1])]
    filtered = tf.gather_nd(tf.transpose(matrix), columns)
    return tf.transpose(filtered)

def extract_targets(features, target_dims):
    return extract_columns(features, target_dims)

def extract_inputs(features, target_dims):
    num_features = int(tf.shape(features)[1])
    target_dims = reindex(target_dims, num_features)
    input_dims = list(set(range(num_features)).difference(set(target_dims)))
    return extract_columns(features, input_dims)

def to_categorical(vector):
    assert len(tf.shape(vector)) == 1
    unique, idx = tf.unique(vector)
    return idx, tf.shape(unique)[0]

def shuffle_and_split(x, y, split_ratio):
    len_x = int(tf.shape(x)[0])
    len_y = int(tf.shape(y)[0])
    assert len_x == len_y
    indices = tf.range(start=0, limit=len_x, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_x = tf.gather(x, shuffled_indices)
    shuffled_y = tf.gather(y, shuffled_indices)
    train_lim = int(split_ratio * len_x)
    x_train = shuffled_x[train_lim:,:]
    y_train = shuffled_y[train_lim:]
    x_test = shuffled_x[:train_lim,:]
    y_test = shuffled_y[:train_lim]
    return (x_train, y_train), (x_test, y_test)
