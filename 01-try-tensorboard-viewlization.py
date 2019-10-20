import numpy as np
import tensorflow as tf

a = tf.constant(3,name='a')
b = tf.constant(6,name='b')

sum = tf.add(a,b,name='aadd')

var = tf.Variable(10,name='var')

var_initer = tf.global_variables_initializer()

normal = tf.random_normal([2,3])

with tf.Session() as sess:
    filewrite = tf.summary.FileWriter(r'C:\Users\jiujue\PycharmProjects\try-deep-Learning\temp',graph=sess.graph)

    print(sess.run([var_initer,normal,sum]))


