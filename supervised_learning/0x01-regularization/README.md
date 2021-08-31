# Neural Network Regularization Project

## Covers L2 regularization, dropout gradient descent and early stopping

* l2_reg_cost(cost, lambtha, weights, L, m):

> Calculates the cost of a neural network with L2 regularization.

* l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):

> Updates the weights and biases of a neural network using gradient descent with L2 regularization.

* l2_reg_cost(cost):

> Calculates the cost of a neural network with L2 regularization.

* l2_reg_create_layer(prev, n, activation, lambtha):

> Creates a tensorflow layer that includes L2 regularization.

* dropout_forward_prop(X, weights, L, keep_prob):

> Conducts forward propagation using Dropout.

* dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):

> Updates the weights of a neural network with Dropout regularization using gradient descent.

* dropout_create_layer(prev, n, activation, keep_prob):

> Creates a layer of a neural network using dropout.

* early_stopping(cost, opt_cost, threshold, patience, count):

> Determines if you should stop gradient descent early.