#!/usr/bin/env python3
"""Module contains train_mini_batch() function"""

import tensorflow as tf
import numpy as np
from tensorflow.python.training.input import batch

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
       Trains a loaded neural network model using
       mini-batch gradient descent

       Args:
         X_train: numpy.ndarray - (m, 784) containing training data
         Y_train: numpy.ndarray - (m, 10) one-hot containing traing labels
         X_valid: numpy.ndarray - (m, 784) containing validation data
         Y_valid: numpy.ndarray - (m, 10) one-hot containing validation labels
         batch_size: number of data points in a batch
         epochs: training iteration over entire dataset
         load_path: path from which to load the model
         save_path: path to where the model should be saved after training

       Returns:
         The path where the model was saved.
    """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        for epoch in range(epochs + 1):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            print("After {} epochs:".format(epoch))
            tLoss, tAccuracy = sess.run([loss, accuracy],
                                        feed_dict={x: X_train, y: Y_train})
            vLoss, vAccuracy = sess.run([loss, accuracy],
                                        feed_dict={x: X_valid, y: Y_valid})
            print("\tTraining Cost:", tLoss)
            print("\tTraining Accuracy:", tAccuracy)
            print("\tValidation Cost:", vLoss)
            print("\tValidation Accuracy:", vAccuracy)
            if epoch == epochs:
                break
            for step in range(0, X_train.shape[0], batch_size):
                feed = {
                    x: X_train[step:step+batch_size, :],
                    y: Y_train[step:step+batch_size, :]
                    }
                if int(step/batch_size) % 100 == 0 and step != 0:
                    print("\tStep {}:".format(int(step/batch_size)))
                    mini_loss, mini_acc = sess.run([loss, accuracy],
                                                   feed_dict=feed)
                    print("\t\tCost:", mini_loss)
                    print("\t\tAccuracy:", mini_acc)
                sess.run(train_op, feed_dict=feed)
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
