import tensorflow as tf

n_inputs = 30
n_nodes_hl1 = 500
n_classes = 1
batch_size = 100    # numero de amostras a ser propagadas em cada epoch

x = tf.placeholder('float', [None, n_inputs])
y = tf.placeholder('float')

def model(data):
    hidden_1_layer = {
        'weights':tf.Variable(tf.truncated_normal(shape=[n_inputs, n_nodes_hl1], stddev=1.0)),
        'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))
    }
    output_layer = {
        'weights':tf.Variable(tf.truncated_normal(shape=[n_nodes_hl1, n_classes], stddev=1.0)),
        'biases':tf.Variable(tf.constant(0.1, shape=[n_classes]))
    }

    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) # activation function

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output