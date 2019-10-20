import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_path = r'C:\Users\jiujue\PycharmProjects\try-deep-Learning\data\mnist\input_data'


def convolution():
    """
    convolution neural network
    :return:None
    :rtype:
    """
    mnist = input_data.read_data_sets(data_path, one_hot=True)

    # ready data
    with tf.variable_scope('ready-data'):

        x = tf.placeholder(tf.float32,[None,784])
        y_true = tf.placeholder(tf.int32,[None,10])

    with tf.variable_scope('reshape-data-of-x'):
        reshape_x = tf.reshape(x,[-1,28,28,1])

    # [None, 28, 28, 1]--->[None, 28, 28, 32]--->[None, 14, 14, 32]--->[None, 14, 14, 64]--->[None, 7, 7, 64]
    with tf.variable_scope('make-model'):
        w_conv_1 = tf.Variable(tf.random_normal([5,5,1,32]))
        b_conv_1 = tf.Variable(tf.random_normal([32]))

        # [None, 28, 28, 1] ---> [None, 28, 28, 32]
        conv_relu_1 = tf.nn.relu(tf.nn.conv2d(reshape_x,w_conv_1,[1,1,1,1],padding='SAME') +b_conv_1)
        # [None, 28, 28, 32]---> [None, 14, 14, 32]
        conv_looping_1 = tf.nn.max_pool(conv_relu_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        w_conv_2 = tf.Variable(tf.random_normal([5,5,32,64]))
        b_conv_2 = tf.Variable(tf.random_normal([64]))

        # [None, 14, 14, 32] ---> [None, 14, 14, 64]
        conv_relu_2 = tf.nn.relu(tf.nn.conv2d(conv_looping_1,w_conv_2,[1,1,1,1],padding='SAME') + b_conv_2)
        # [None, 14, 14, 64]---> [None, 7, 7, 64]
        conv_looping_2 = tf.nn.max_pool(conv_relu_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


    with tf.variable_scope('reshape-data-of-conv_looping_2'):
        # [None, 7, 7, 64]-->[None,7*7*64]
        reshape_conv_looping_2 = tf.reshape(conv_looping_2,[-1,7*7*64])

    with tf.variable_scope('full-connected'):

        w_fc = tf.Variable(tf.random_normal([7*7*64,10]))
        b_fc = tf.Variable(tf.random_normal([10]))
        # ( [None,7*7*64]* [7*7*64,10]+ 10 ) -->[None,10]
        y_predict = tf.matmul(reshape_conv_looping_2, w_fc) + b_fc

    with tf.variable_scope('computer-loss'):
        loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    with tf.variable_scope('optimizer'):
        train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss=loss)


    with tf.variable_scope('accuracy-rate'):
        all_acc = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        acc = tf.reduce_mean(tf.cast(all_acc,tf.float32))


    init_op = tf.global_variables_initializer()


    with tf.Session() as sess:

        sess.run(init_op)

        for i in range(20000):
            x_,y_ = mnist.train.next_batch(50)

            sess.run(train_op,feed_dict={x:x_,y_true:y_})

            print('{} th, acc: {}'.format(i,sess.run(acc,feed_dict={x:x_,y_true:y_})))


if __name__ == '__main__':
    convolution()









































