import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn import dynamic_rnn
import numpy as np
from isan_cell import IsanCell


def run_lstm_mnist(lstm_cell=IsanCell, hidden_size=32, batch_size=256, steps=20):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    learning_rate = 0.001
    x = tf.placeholder(tf.float32, (batch_size, 784, 1))
    y_ = tf.placeholder(tf.float32, (batch_size, 10))
    outputs, state = dynamic_rnn(lstm_cell(hidden_size), x, dtype=tf.float32)
    rnn_out = outputs[:, -1, :]

    y = slim.fully_connected(rnn_out,
                             num_outputs=10,
                             activation_fn=None)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    for i in range(steps):
        batch = mnist.train.next_batch(batch_size)
        feed_dict = {x: np.expand_dims(batch[0], axis=2), y_: batch[1]}
        tr_loss, tr_acc, _ = sess.run([cross_entropy, accuracy, grad_update], feed_dict=feed_dict)
        print(tr_loss)


def main():
    run_lstm_mnist(lstm_cell=IsanCell, hidden_size=32, batch_size=128, steps=2000)


if __name__ == '__main__':
    main()
