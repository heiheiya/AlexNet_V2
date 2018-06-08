import tensorflow as tf

def conv_layer(input, ksize, strides, name, b_value=0.0, padding='VALID', group=1):
    with tf.variable_scope(name) as scope:
        channels = int(input.get_shape()[-1])
        filter = tf.get_variable('weights', shape=[ksize[0], ksize[1], channels/group, ksize[3]])
        biases = tf.get_variable('biases', shape=[ksize[3]])

        if group == 1:
            conv = tf.nn.conv2d(input=input, filter=filter, strides=strides, padding=padding)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
            weight_groups = tf.split(axis=3, num_or_size_splits=group, value=filter)
            output_groups = [tf.nn.conv2d(input=i, filter=k, strides=strides, padding=padding) for i, k in zip(input_groups, weight_groups)]
            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        relu = tf.nn.relu(bias, name=scope.name)
        print_activations(relu)
        return relu


def max_pool_layer(input, ksize, strides, name, padding='VALID'):
    max_pool = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding, name=name)
    print_activations(max_pool)
    return max_pool

def full_connected_layer(input, n_out, name, b_value =0.0, relu=True):
    shape = input.get_shape().as_list()
    dim = 1
    for d in range(len(shape)-1):
        dim *= shape[d+1]
    x = tf.reshape(input, [-1, dim])
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[n_in, n_out])
        biases = tf.get_variable('biases', shape=[n_out])
        if relu == True:
            fc = tf.nn.relu(tf.add(tf.matmul(x, weights), biases), name=scope.name)
        else:
            fc = tf.add(tf.matmul(x, weights), biases)
        print_activations(fc)
        return fc

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())