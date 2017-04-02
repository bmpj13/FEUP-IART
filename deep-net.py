# YouTube - sentdex (Neural Networks)


'''
input > weights > hidden layer 1 (activation function) > weights
> hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost (or loss) function (p.ex. cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer..... SGD, Adagrad)

backpropagation

feed forward + backprop = epoch
'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 10 classes, 0-9
'''
0 = [1, 0, 0, 0, ..., 0] (10 de tamanho)
1 = [0, 1, 0, 0, ..., 0]
2 = [0, 0, 1, 0, ..., 0]
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100    # numero de amostras a ser propagadas

# height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):

    # NOTA: BIASES PROVAVELMENTE ERRADAS (NAO USAR random_normal)
    hidden_1_layer = {
        'weights':tf.Variable(tf.truncated_normal(shape=[784, n_nodes_hl1], stddev=1.0)),
        'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))
    }
    hidden_2_layer = {
        'weights':tf.Variable(tf.truncated_normal(shape=[n_nodes_hl1, n_nodes_hl2], stddev=1.0)),
        'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl2]))
    }
    hidden_3_layer = {
        'weights':tf.Variable(tf.truncated_normal(shape=[n_nodes_hl2, n_nodes_hl3], stddev=1.0)),
        'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl3]))
    }
    output_layer = {
        'weights':tf.Variable(tf.truncated_normal(shape=[n_nodes_hl3, n_classes], stddev=1.0)),
        'biases':tf.Variable(tf.constant(0.1, shape=[n_classes]))
    }

    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) # activation function

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2) # activation function

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3) # activation function

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)     # learning_rate = 0.001

    hm_epochs = 50      # cycle feedforward + backpropagation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(hm_epochs):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)