#!/usr/bin/env python3
"""
   Module contains
   def model(Data_train, Data_valid, layers,
   activations, alpha=0.001, beta1=0.9, beta2=0.999,
   epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
   save_path='/tmp/model.ckpt'):
"""


import tensorflow as tf
import numpy as np


def cost(Y, A):
    loss = - (Y * tf.math.log(A) + (1 - Y) * tf.math.log(1 - A))
    cost = np.sum(loss) / Y.shape[0]
    return cost


def calculate_accuracy(y, y_pred):
    """Calculates accuracy of prediction"""
    pred = tf.math.argmax(y_pred, axis=1)
    ny = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(pred, ny)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


def create_batch_norm_layer(prev, n, activation):
    """
       Creates a batch normalization layer for a neural network in tensorflow.

       Args:
         prev is the activated output of the previous layer
         n: number of nodes in the layer to be created
         activation: activation function that should be used on the output
           of the layer

       Returns:
         A tensor of the activated output for the layer.
    """
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layerX = tf.layers.dense(prev, n, kernel_initializer=weights)
    mean, variance = tf.nn.moments(layerX, 0)
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(
        layerX, mean, variance, beta, gamma, epsilon
    )
    if activation is not None:
        return activation(batch_norm)
    return batch_norm


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward_prop using tensorflow"""
    pred = x
    for i in range(len(layer_sizes)):
        pred = create_batch_norm_layer(pred, layer_sizes[i], activations[i])
    return pred


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
       Builds, trains, and saves a neural network model in tensorflow
       using Adam optimization, mini-batch gradient descent, learning
       rate decay, and batch normalization.

       Args:
         Data_train: tuple containing the training inputs and
           training labels, respectively
         Data_valid: tuple containing the validation inputs and
           validation labels, respectively
         layers: list containing the number of nodes in each layer of
           the network
         activation: list containing the activation functions used for
           each layer of the network
         alpha: learning rate
         beta1: weight for the first moment of Adam Optimization
         beta2: weight for the second moment of Adam Optimization
         epsilon: small number used to avoid division by zero
         decay_rate: decay rate for inverse time decay of the learning rate
         batch_size: number of data points that should be in a mini-batch
         epochs: number of times the training should pass through
           the whole dataset
         save_path: path where the model should be saved to

       Returns:
         The path where the model was saved.
    """
    X_train, Y_train = Data_train[0], Data_train[1]
    X_valid, Y_valid = Data_valid[0], Data_valid[1]
    # alphaP = tf.placeholder(name="alpha", dtype=tf.float32)
    global_step = tf.Variable(0, trainable=False)
    # decay_step = tf.placeholder(name="decay_step", dtype=tf.float32)
    decay_step = X_train.shape[1] // batch_size
    x = tf.placeholder(name="x", dtype=tf.float32,
                       shape=[None, X_train.shape[1]])
    y = tf.placeholder(name="y", dtype=tf.float32,
                       shape=[None, Y_train.shape[1]])
    # tf.add_to_collection("alpha", alphaP)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    train_op = tf.train.AdamOptimizer(
        alpha, beta1, beta2, epsilon
        ).minimize(loss, global_step)
    alpha = tf.train.inverse_time_decay(
                                  alpha, global_step, decay_step,
                                  decay_rate, staircase=True
                                  )
    tf.add_to_collection("train_op", train_op)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        nx = X_train.shape[0]
        batches = nx // batch_size
        if batches % batch_size != 0:
            batches += 1
        for epoch in range(epochs + 1):
            tLoss = loss.eval({x: X_train, y: Y_train})
            tAccuracy = accuracy.eval({x: X_train, y: Y_train})
            vLoss = loss.eval({x: X_valid, y: Y_valid})
            vAccuracy = accuracy.eval({x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(tLoss))
            print("\tTraining Accuracy: {}".format(tAccuracy))
            print("\tValidation Cost: {}".format(vLoss))
            print("\tValidation Accuracy: {}".format(vAccuracy))
            if epoch == epochs:
                break
            shuff = np.random.permutation(len(X_train))
            X_shuff, Y_shuff = X_train[shuff], Y_train[shuff]
            for step in range(batches):
                feed = {
                    x: X_shuff[batch_size*step:batch_size*(step+1)],
                    y: Y_shuff[batch_size*step:batch_size*(step+1)]
                    }
                # alpha = alpha_decay_op.eval(feed)
                sess.run(train_op, feed)
                if (step+1) % 100 == 0 and step != 0:
                    print("\tStep {}:".format(step+1))
                    mini_loss, mini_acc = loss.eval(feed), accuracy.eval(feed)
                    print("\t\tCost: {}".format(mini_loss))
                    print("\t\tAccuracy: {}".format(mini_acc))
            # alpha = alpha_decay_op.eval(feed)
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
