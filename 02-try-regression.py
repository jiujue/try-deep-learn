import numpy as np
import tensorflow as tf
import os

def regression():
    '''
    myself regression
    :return: None
    '''
    # ready data
    with tf.variable_scope('ready-data'):
        x = tf.random.normal([100,1],stddev=3.7,mean=5)
        y_true = tf.matmul(x,[[0.7]]) + 0.8

    # make model
    #   make variable
    with tf.variable_scope('make-model'):
        weight = tf.Variable(tf.random.normal([1,1],mean=0.0,stddev=0.1),name='w')
        bias = tf.Variable(0.0,name='b')
        #   init Variable


        y_predict = tf.matmul(x,weight) + bias

    with tf.variable_scope('calc-loss'):
        # calculation less
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope('optimizer'):
        # gradient decent
        train_op = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(loss)

    tf.compat.v1.summary.scalar('loss',loss)
    tf.compat.v1.summary.histogram('weight',weight)

    merged = tf.compat.v1.summary.merge_all()

    init_op = tf.compat.v1.global_variables_initializer()

    saver = tf.compat.v1.train.Saver()



    # start session and run
    with  tf.compat.v1.Session() as sess:
        sess.run(init_op)

        filewrite = tf.compat.v1.summary.FileWriter('./temp',sess.graph)


        print('initialize data : w ,bias ',weight.eval(),bias.eval())

        if os.path.exists('./temp/model/checkpoint'):
            saver.restore(sess,'./temp/model/model')

        for i in range(100):
            sess.run(train_op)
            # write in variable change
            summary = sess.run(merged)
            filewrite.add_summary(summary,i)
            # print variable change
            print('{i} th training result:{w} {b}'.format(i=i,w=weight.eval(),b=bias.eval()))

        # save model
        saver.save(sess,'./temp/model/model')



    # save model

if __name__ == '__main__':
    regression()
