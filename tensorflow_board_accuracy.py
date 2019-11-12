# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('datasets', one_hot=True)

batch_size = 100
m_batch = mnist.train.num_examples//batch_size

def weight_variable(input_shape):
    initial = tf.truncated_normal(input_shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(input_shape):
    initial = tf.constant(0.1, shape=input_shape)
    return tf.Variable(initial)


def conv2d(input, weight):
    # input为4维数组，1表示batch_size，2表示输出的高度（in_height），
    # 3表示输出的宽度(in_weight)，最后1维表示输入图片的通道数(channel)
    # conv2d的第二个参数表示filter卷积核，filter[0]表示卷积核的高度, filter[1]表示
    # 卷积核的宽度，filter[2]表示输入的通道数(in_channel)，filter[3]表示输出的通道数
    # strides[0]=strides[3]=1，并且永远为1，strides[1]表示x方向的步长，strides[2]表示
    # y方向的步长
    return tf.nn.conv2d(
        input, filter=weight, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1_layer'):
    with tf.name_scope('w_conv1'):
        w_conv1 = weight_variable([5, 5, 1, 32])
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32])

    with tf.name_scope('conv1_conv_op'):
        # 本身输入为28*28*1，卷积核为5*5,输出通道为32，padding，因此，输出为28*28*32
        h1_conv1 = conv2d(x_image, w_conv1) + b_conv1
    with tf.name_scope('conv1_relu_op'):
        h_conv1 = tf.nn.relu(h1_conv1)
    with tf.name_scope('conv1_pool_op'):
        # 池化为2*2, 步长为2，pandding，输出为14*14*32
        h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('conv2_layer'):
    with tf.name_scope('w_conv2'):
        w_conv2 = weight_variable([5, 5, 32, 64])
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64])

    with tf.name_scope('conv2_conv_op'):
        # 输入为14*14*32,卷积核为5*5,输出通道为64，panding,因此，输出为14*14*64
        h2_conv2 = conv2d(h_pool1, w_conv2) + b_conv2
    with tf.name_scope('conv2_relu_op'):
        h_conv2 = tf.nn.relu(h2_conv2)
    with tf.name_scope('conv2_pool_op'):
        # 再进行步长为2的pool，输出结果为7*7*64
        h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1_layer'):
    with tf.name_scope('w_fc1'):
        w_fc1 = weight_variable([7*7*64, 1024])
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024])
    with tf.name_scope('fc1_flatten_op'):
        # 将4维输出扁平化
        h_pool2_flatten = tf.reshape(h_pool2, [-1, 7*7*64])
    with tf.name_scope('fc1_matmul_op'):
        h_fc1 = tf.matmul(h_pool2_flatten, w_fc1) + b_fc1
    with tf.name_scope('fc1_relu_op'):
        h_fc1 = tf.nn.relu(h_fc1)
    with tf.name_scope('keep_prob_op'):
        keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('dropout_op'):
        h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

with tf.name_scope('fc2_layer'):
    with tf.name_scope('w_fc2'):
        w_fc2 = weight_variable([1024, 10])
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10])
    with tf.name_scope('fc2_matmul_op'):
        fc2 = tf.matmul(h_fc1_dropout, w_fc2) + b_fc2
    with tf.name_scope('fc2_softmax_op'):
        prediction = tf.nn.softmax(fc2)

with tf.name_scope('corss_entropy'):
    corss_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('corss_entropy', corss_entropy)

with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(1e-4).minimize(corss_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('models/train', sess.graph)
    test_writer = tf.summary.FileWriter('models/test', sess.graph)
    for epoch in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run([train], feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
        _summary = sess.run(
            merged, feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        train_writer.add_summary(_summary, epoch)
        batch_tx, batch_ty = mnist.test.next_batch(batch_size)
        __summary = sess.run(
            merged, feed_dict={x:batch_tx,y:batch_ty, keep_prob:1.0})
        test_writer.add_summary(__summary, epoch)
        if epoch%100 == 0:
            test_acc = sess.run(
                accuracy, feed_dict={x:mnist.test.images,
                                     y:mnist.test.labels, keep_prob:1.0})
            train_acc = sess.run(
                accuracy, feed_dict={x:mnist.train.images[:10000],
                                     y:mnist.train.labels[:10000],
                                     keep_prob:1.0})
            print('%d test accuracy %f, train_acc %f' % (
                epoch, test_acc, train_acc))