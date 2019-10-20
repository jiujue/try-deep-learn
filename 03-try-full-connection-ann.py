import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import  inception_v3_base
import os



def full_connection():

    data_path = r'C:\Users\jiujue\PycharmProjects\try-deep-Learning\data\mnist\input_data'
    mnist = input_data.read_data_sets(data_path, one_hot=True)

    # ready data
    with tf.variable_scope('ready-data'):
        x = tf.placeholder(tf.float32,[None,784])
        y_true = tf.placeholder(tf.int32,[None,10])

    # made model
    with tf.variable_scope('model'):
        weight = tf.Variable(tf.random_normal([784,10]))
        # bias = tf.Variable(tf.random_normal([10]))
        bias = tf.Variable(tf.constant(0.0, shape=[10]))


    # computer loss
    with tf.variable_scope('computer-losses'):
        y_predict = tf.matmul(x, weight) + bias
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict)

    # use gradient descent optimizer
    with tf.variable_scope('optimizer'):
        optimizer_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.variable_scope('acc'):
        all_acc = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        acc = tf.reduce_mean(tf.cast(all_acc,tf.float32))

    # initialize variable
    init_op = tf.global_variables_initializer()

    # collection variable
    tf.summary.histogram('losses',loss)

    tf.summary.scalar("acc", acc)

    tf.summary.histogram('weight',weight)
    tf.summary.histogram('bias',bias)

    # merge variable
    merged = tf.summary.merge_all()

    # create saver
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # init variable
        sess.run(init_op)

        # create file writer

        write = tf.summary.FileWriter('./temp/summary/',graph=sess.graph)

        for i in range(2000):
            # obtain data from mnist
            x_, y_ = mnist.train.next_batch(50)

            sess.run(optimizer_op,feed_dict={x: x_, y_true: y_})

            print( '%d th ,acc: %f'%(i,sess.run(acc ,feed_dict={x: x_, y_true: y_}) ))

            summary = sess.run(merged,feed_dict={x: x_, y_true: y_})
            write.add_summary(summary,i)

            # print(dir(loss))
            # print(help(loss))

        saver.save(sess,'./temp/model/full-connection')



if __name__ == '__main__':
    full_connection()
