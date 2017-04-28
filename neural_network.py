import tensorflow as tf
import parser
import numpy as np
from sys import argv, exit

n_inputs = 30
n_nodes_hl1 = None # = 500
n_classes = 1

train_set_size = None # = 4000
batch_size = None # = 100    # numero de amostras a ser propagadas em cada epoch
epochs = None # = 30         # ciclos feedforward + backpropagation

x = tf.placeholder('float', [None, n_inputs])
y = tf.placeholder('float')

def read_inputs():
    if len(argv)<4:
        wrong_input='''Arguments: neural_network.py <file_path> <n_nodes_hl1> <train_set_size> <batch_size> <epochs>'''
        print(wrong_input)
        exit()
    else:
        if parser.setFilePath(argv[1]):
            global n_nodes_hl1,train_set_size,batch_size,epochs
            n_nodes_hl1=int(argv[2])
            train_set_size=int(argv[3])
            batch_size=int(argv[4])
            epochs=int(argv[5])
        else:
            invalid_path='''Arguments: invalid <file_path>'''
            print(invalid_path)
            exit()

read_inputs()

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
    return output


def train(x):
    output = model(x)
    prediction = tf.nn.sigmoid(output)
    predicted_class = tf.greater(prediction, 0.5)
    cost = tf.nn.l2_loss(prediction-y)
    optimizer = tf.train.AdamOptimizer().minimize(cost)     # learning_rate = 0.001
    correct = tf.equal(predicted_class, tf.equal(y, 1.0))
    accuracy = tf.reduce_mean( tf.cast(correct, 'float'))
    weights=tf.Variable(tf.truncated_normal(shape=[n_inputs, n_nodes_hl1], stddev=1.0))
    bias=tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        activation_summary = tf.summary.histogram("output", prediction)
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        cost_summary = tf.summary.scalar("cost", cost)
        weight_summary=tf.summary.histogram("weights",weights.eval(session=sess))
        bias_summary=tf.summary.histogram("biases",bias.eval(session=sess))
        all_summary = tf.summary.merge_all()

        writer = tf.summary.FileWriter("output", sess.graph)

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(train_set_size/batch_size):
                epoch_x, epoch_y = parser.next_training_set(batch_size=batch_size)

                if i % 10 == 0:
                    summary_results, _, c = sess.run([all_summary, optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    writer.add_summary(summary_results, i)
                    epoch_loss += c
                else:
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

            print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)
            parser.reset_batch_start()


        test_x, test_y = parser.test_set(train_set_size)
        print('Accuracy:', accuracy.eval({x:test_x, y:test_y}) * 100)


train(x)
