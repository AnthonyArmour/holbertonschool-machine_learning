# 0x01-Optimization Project

In this project we create to learn about different optimization techniques for deep neural networks.

## Optimization Tools Made

> normalization_constants(X):

-- Calculates the normalization (standardization) constants of a matrix.

> normalize(X, m, s):

-- Normalizes a matrix:

> shuffle_data(X, Y):

-- Shuffles the data points in two matrices the same way.

> train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt")

-- Trains a loaded neural network model using mini-batch gradient descent.

> moving_average(data, beta):

-- Calculates the weighted moving average of a data set.

> update_variables_momentum(alpha, beta1, var, grad, v):

-- The updated variable and the new moment, respectively.

> create_momentum_op(loss, alpha, beta1):

-- Creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm

> update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):

-- Updates a variable using the RMSProp optimization algorithm.

> create_RMSProp_op(loss, alpha, beta2, epsilon):

-- Creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm

> update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):

-- Updates a variable in place using the Adam optimization algorithm.

> create_Adam_op(loss, alpha, beta1, beta2, epsilon):

-- Creates the training operation for a neural network in tensorflow using the Adam optimization algorithm.

> learning_rate_decay(alpha, decay_rate, global_step, decay_step):

-- Updates the learning rate using inverse time decay in numpy.

> learning_rate_decay(alpha, decay_rate, global_step, decay_step):

-- Creates a learning rate decay operation in tensorflow using inverse time decay.

> batch_norm(Z, gamma, beta, epsilon):

-- Normalizes an unactivated output of a neural network using batch normalization

> create_batch_norm_layer(prev, n, activation):

-- Creates a batch normalization layer for a neural network in tensorflow.

> model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):

-- Builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization