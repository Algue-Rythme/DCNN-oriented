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
