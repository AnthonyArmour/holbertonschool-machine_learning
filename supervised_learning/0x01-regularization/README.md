[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Neural Network Regularization Project


## Tasks
Covers L2 regularization, dropout gradient descent, and early stopping.

## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| tensorflow         | ^2.6.0  |


### [l2_reg_cost](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-regularization/0-l2_reg_cost.py "l2_reg_cost")
Calculates the cost of a neural network with L2 regularization.
``` python
#!/usr/bin/env python3

import numpy as np
l2_reg_cost = __import__('0-l2_reg_cost').l2_reg_cost

if __name__ == '__main__':
    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['W2'] = np.random.randn(128, 256)
    weights['W3'] = np.random.randn(10, 128)

    cost = np.abs(np.random.randn(1))

    print(cost)
    cost = l2_reg_cost(cost, 0.1, weights, 3, 1000)
    print(cost)
```

### [l2_reg_gradient_descent](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-regularization/1-l2_reg_gradient_descent.py "l2_reg_gradient_descent")
Updates the weights and biases of a neural network using gradient descent with L2 regularization.
``` python
#!/usr/bin/env python3

import numpy as np
l2_reg_gradient_descent = __import__('1-l2_reg_gradient_descent').l2_reg_gradient_descent


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    cache = {}
    cache['A0'] = X_train
    cache['A1'] = np.tanh(np.matmul(weights['W1'], cache['A0']) + weights['b1'])
    cache['A2'] = np.tanh(np.matmul(weights['W2'], cache['A1']) + weights['b2'])
    Z3 = np.matmul(weights['W3'], cache['A2']) + weights['b3']
    cache['A3'] = np.exp(Z3) / np.sum(np.exp(Z3), axis=0)
    print(weights['W1'])
    l2_reg_gradient_descent(Y_train_oh, weights, cache, 0.1, 0.1, 3)
    print(weights['W1'])
```

### [l2_reg_cost](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-regularization/2-l2_reg_cost.py "l2_reg_cost")
Calculates the cost of a neural network with L2 regularization.
``` python
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    oh = np.zeros((classes, m))
    oh[Y, np.arange(m)] = 1
    return oh

np.random.seed(4)
m = np.random.randint(1000, 2000)
c = 10
lib= np.load('../data/MNIST.npz')

X = lib['X_train'][:m].reshape((m, -1))
Y = one_hot(lib['Y_train'][:m], c).T

n0 = X.shape[1]
n1, n2 = np.random.randint(10, 1000, 2)

lam = np.random.uniform(0.01)
tf.set_random_seed(0)

x = tf.placeholder(tf.float32, (None, n0))
y = tf.placeholder(tf.float32, (None, c))

a1 = tf.layers.Dense(n1, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), kernel_regularizer=tf.contrib.layers.l2_regularizer(lam))(x)
a2 = tf.layers.Dense(n2, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), kernel_regularizer=tf.contrib.layers.l2_regularizer(lam))(a1)
y_pred = tf.layers.Dense(c, activation=None, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), kernel_regularizer=tf.contrib.layers.l2_regularizer(lam))(a2)

cost = tf.losses.softmax_cross_entropy(y, y_pred)

l2_cost = l2_reg_cost(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(l2_cost, feed_dict={x: X, y: Y}))
```

### [l2_reg_create_layer](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-regularization/3-l2_reg_create_layer.py "l2_reg_create_layer")
Creates a tensorflow layer that includes L2 regularization.
``` python
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost
l2_reg_create_layer = __import__('3-l2_reg_create_layer').l2_reg_create_layer

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((m, classes))
    one_hot[np.arange(m), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)

    tf.set_random_seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    h1 = l2_reg_create_layer(x, 256, tf.nn.tanh, 0.1)
    y_pred = l2_reg_create_layer(x, 10, None, 0.)
    cost = tf.losses.softmax_cross_entropy(y, y_pred)
    l2_cost = l2_reg_cost(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(l2_cost, feed_dict={x: X_train, y: Y_train_oh}))
```

### [dropout_forward_prop](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-regularization/4-dropout_forward_prop.py "dropout_forward_prop")
Conducts forward propagation using Dropout.
``` python
#!/usr/bin/env python3

import numpy as np
dropout_forward_prop = __import__('4-dropout_forward_prop').dropout_forward_prop


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    cache = dropout_forward_prop(X_train, weights, 3, 0.8)
    for k, v in sorted(cache.items()):
        print(k, v)
```

### [dropout_gradient_descent](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-regularization/5-dropout_gradient_descent.py "dropout_gradient_descent")
Updates the weights of a neural network with Dropout regularization using gradient descent.
``` python
#!/usr/bin/env python3

import numpy as np
dropout_forward_prop = __import__('4-dropout_forward_prop').dropout_forward_prop
dropout_gradient_descent = __import__('5-dropout_gradient_descent').dropout_gradient_descent


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    cache = dropout_forward_prop(X_train, weights, 3, 0.8)
    print(weights['W2'])
    dropout_gradient_descent(Y_train_oh, weights, cache, 0.1, 0.8, 3)
    print(weights['W2'])
```

### [dropout_create_layer]( https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-regularization/6-dropout_create_layer.py"dropout_create_layer")
Creates a layer of a neural network using dropout.
``` python
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
dropout_create_layer = __import__('6-dropout_create_layer').dropout_create_layer

if __name__ == '__main__':
    tf.set_random_seed(0)
    np.random.seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    X = np.random.randint(0, 256, size=(10, 784))
    a = dropout_create_layer(x, 256, tf.nn.tanh, 0.8)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a, feed_dict={x: X}))
```

### [early_stopping](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-regularization/7-early_stopping.py "early_stopping")
Determines if you should stop gradient descent early.
``` python
#!/usr/bin/env python3

early_stopping = __import__('7-early_stopping').early_stopping

if __name__ == '__main__':
    print(early_stopping(1.0, 1.9, 0.5, 15, 5))
    print(early_stopping(1.1, 1.5, 0.5, 15, 2))
    print(early_stopping(1.0, 1.5, 0.5, 15, 8))
    print(early_stopping(1.0, 1.5, 0.5, 15, 14))
```

```
(False, 0)
(False, 3)
(False, 9)
(True, 15)
```