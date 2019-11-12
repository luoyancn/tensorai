# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


mnist = input_data.read_data_sets('datasets', one_hot=True)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean) # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev) ## 标准差
        tf.summary.scalar('max', tf.reduce_max(var)) ## 最大值
        tf.summary.scalar('min', tf.reduce_min(var)) ## 最小值
        tf.summary.histogram('histogram', var) ## 直方图

image_num = 3000
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]),
                        trainable=False, name='embedding')

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name='input_x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='input_y')

with tf.name_scope('input_reshaped'):
    input_reshaped = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input_reshaped', input_reshaped, 10)

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        w = tf.Variable(tf.zeros(shape=[784, 10]), name='weight_1')
        variable_summaries(w)
    with tf.name_scope('bias'):
        b = tf.Variable(tf.zeros(shape=[10], name='bias_1'))
        variable_summaries(w)
    with tf.name_scope('matmul'):
        r = tf.matmul(x, w) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(r)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, axis=1),
                                      tf.argmax(prediction, axis=1))

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('loss', accuracy)

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    if tf.gfile.Exists('models/metadata.tsv'):
        tf.gfile.DeleteRecursively('models/metadata.tsv')
    with open('models/metadata.tsv', 'w') as f:
        labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
        for i in range(image_num):
            f.write(str(labels[i]) + '\n')

    projector_writer = tf.summary.FileWriter('models', sess.graph)
    saver = tf.train.Saver()

    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = 'models/metadata.tsv'
    embed.sprite.image_path = 'models/mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(projector_writer, config)

    for i in range(200):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _, _summary = sess.run(
            [train, merged], feed_dict={x:batch_xs, y: batch_ys},
            options=run_options, run_metadata=run_metadata)
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        projector_writer.add_summary(_summary, i)
        acc_val = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('%d epoch, Test accuracy %f' % (i, acc_val))
    saver.save(sess, 'models/model.ckpt', global_step=10)
    projector_writer.close()