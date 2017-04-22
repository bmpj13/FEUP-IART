import tensorflow as tf
import parser
import numpy as np

n_inputs = 30
n_nodes_hl1 = 500
n_classes = 1

train_set_size = 4000
batch_size = 100    # numero de amostras a ser propagadas em cada epoch
epochs = 20         # ciclos feedforward + backpropagation

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
    l1 = tf.nn.sigmoid(l1) # activation function

    output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])
    return tf.transpose(output)


def train(x):
    prediction = model(x)
    activation = tf.nn.sigmoid(prediction)
    cost = tf.nn.l2_loss(activation-y)
    optimizer = tf.train.AdamOptimizer().minimize(cost)     # learning_rate = 0.001

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(train_set_size/batch_size):
                epoch_x = []
                epoch_y = []
                rows = parser.next_training_set(batch_size=batch_size)
                for row in rows:
                    epoch_x.append(row[:-1])
                    epoch_y.append(row[-1])

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)
            parser.reset_batch_start()

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        activation_summary = tf.summary.histogram("output", activation)
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        cost_summary = tf.summary.scalar("cost", accuracy)
        all_summary = tf.summary.merge_all()


train(x)
