#!/usr/bin/env python3
"""Evaluation method for classification model"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluation method for classification model"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        ops = tf.get_collection('train_ops')
        f = {ops[0]: X, ops[1]: Y}
        return sess.run([ops[2], ops[4], ops[3]], feed_dict=f)
