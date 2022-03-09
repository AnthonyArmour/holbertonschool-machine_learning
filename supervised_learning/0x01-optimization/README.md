[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# 0x01-Optimization Project

In this project we create to learn about different optimization techniques for deep neural networks.

## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |
| tensorflow         | ^2.6.0  |
| keras              | ^2.6.0  |

## Optimization Tools Made

### [normalization_constants](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/0-norm_constants.py "normalization_constants")
Calculates the normalization (standardization) constants of a matrix.
``` python
#!/usr/bin/env python3

import numpy as np
normalization_constants = __import__('0-norm_constants').normalization_constants

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(m)
    print(s)
```

### [normalize](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/1-normalize.py "normalize")
Normalizes a matrix.
``` python
#!/usr/bin/env python3

import numpy as np
normalization_constants = __import__('0-norm_constants').normalization_constants
normalize = __import__('1-normalize').normalize

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(X[:10])
    X = normalize(X, m, s)
    print(X[:10])
    m, s = normalization_constants(X)
    print(m)
    print(s)
```

### [shuffle_data](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/2-shuffle_data.py "shuffle_data")
Shuffles the data points in two matrices the same way.
``` python
#!/usr/bin/env python3

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data

if __name__ == '__main__':
    X = np.array([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8], 
                [9, 10]])
    Y = np.array([[11, 12],
                [13, 14],
                [15, 16],
                [17, 18],
                [19, 20]])

    np.random.seed(0)
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    print(X_shuffled)
    print(Y_shuffled)
```

### [train_mini_batch](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/3-mini_batch.py "train_mini_batch")
Trains a loaded neural network model using mini-batch gradient descent.
``` python
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
train_mini_batch = __import__('3-mini_batch').train_mini_batch

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]
    alpha = 0.01
    iterations = 5000

    np.random.seed(0)
    save_path = train_mini_batch(X_train, Y_train_oh, X_valid, Y_valid_oh,
                                 epochs=10, load_path='./graph.ckpt',
                                 save_path='./model.ckpt')
    print('Model saved in path: {}'.format(save_path))
```

### [moving_average](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/4-moving_average.py "moving_average")
Calculates the weighted moving average of a data set.
``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
moving_average = __import__('4-moving_average').moving_average

if __name__ == '__main__':
        data = [72, 78, 71, 68, 66, 69, 79, 79, 65, 64, 66, 78, 64, 64, 81, 71, 69,
                65, 72, 64, 60, 61, 62, 66, 72, 72, 67, 67, 67, 68, 75]
        days = list(range(1, len(data) + 1))
        m_avg = moving_average(data, 0.9)
        print(m_avg)
        plt.plot(days, data, 'r', days, m_avg, 'b')
        plt.xlabel('Day of Month')
        plt.ylabel('Temperature (Fahrenheit)')
        plt.title('SF Maximum Temperatures in October 2018')
        plt.legend(['actual', 'moving_average'])
        plt.show()
```
---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/images/moving-average.png)
---

### [update_variables_momentum](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/5-momentum.py "update_variables_momentum")
The updated variable and the new moment, respectively.
``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
update_variables_momentum = __import__('5-momentum').update_variables_momentum

def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m

    return cost

if __name__ == '__main__':
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y'].T
    X = X_3D.reshape((X_3D.shape[0], -1))

    nx = X.shape[1]
    np.random.seed(0)
    W = np.random.randn(nx, 1)
    b = 0
    dW_prev = np.zeros((nx, 1))
    db_prev = 0
    for i in range(1000):
        A = forward_prop(X, W, b)
        if not (i % 100):
            cost = calculate_cost(Y, A)
            print('Cost after {} iterations: {}'.format(i, cost))
        dW, db = calculate_grads(Y, A, W, b)
        W, dW_prev = update_variables_momentum(0.01, 0.9, W, dW, dW_prev)
        b, db_prev = update_variables_momentum(0.01, 0.9, b, db, db_prev)
    A = forward_prop(X, W, b)
    cost = calculate_cost(Y, A)
    print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.where(A >= 0.5, 1, 0)
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i, 0]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
```

```
Cost after 0 iterations: 4.365105010037203
Cost after 100 iterations: 0.5729736703124042
Cost after 200 iterations: 0.2449357405113111
Cost after 300 iterations: 0.17711325087582164
Cost after 400 iterations: 0.14286111618067307
Cost after 500 iterations: 0.12051674907075896
Cost after 600 iterations: 0.10450664363662196
Cost after 700 iterations: 0.09245615061035156
Cost after 800 iterations: 0.08308760082979068
Cost after 900 iterations: 0.07562924162824029
Cost after 1000 iterations: 0.0695782354732263
```

---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/images/momentum.png)
---

### [create_momentum_op](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/6-momentum.py "create_momentum_op")
Creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm.
``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
create_momentum_op = __import__('6-momentum').create_momentum_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        train_op = create_momentum_op(loss, 0.01, 0.9)
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            if not (i % 100):
                cost = sess.run(loss, feed_dict={x:X, y:Y_oh})
                print('Cost after {} iterations: {}'.format(i, cost))
            sess.run(train_op, feed_dict={x:X, y:Y_oh})
        cost, Y_pred_oh = sess.run((loss, y_pred), feed_dict={x:X, y:Y_oh})
        print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
```

```
Cost after 0 iterations: 2.8232274055480957
Cost after 100 iterations: 0.356641948223114
Cost after 200 iterations: 0.29699304699897766
Cost after 300 iterations: 0.26470813155174255
Cost after 400 iterations: 0.24141179025173187
Cost after 500 iterations: 0.22264979779720306
Cost after 600 iterations: 0.20677044987678528
Cost after 700 iterations: 0.19298051297664642
Cost after 800 iterations: 0.18082040548324585
Cost after 900 iterations: 0.16998952627182007
Cost after 1000 iterations: 0.1602744460105896
```

---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/images/momentum2.png)
---

### [update_variables_RMSProp](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/7-RMSProp.py "update_variables_RMSProp")
Updates a variable using the RMSProp optimization algorithm.
``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
update_variables_RMSProp = __import__('7-RMSProp').update_variables_RMSProp

def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m

    return cost

if __name__ == '__main__':
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y'].T
    X = X_3D.reshape((X_3D.shape[0], -1))

    nx = X.shape[1]
    np.random.seed(0)
    W = np.random.randn(nx, 1)
    b = 0
    dW_prev = np.zeros((nx, 1))
    db_prev = 0
    for i in range(1000):
        A = forward_prop(X, W, b)
        if not (i % 100):
            cost = calculate_cost(Y, A)
            print('Cost after {} iterations: {}'.format(i, cost))
        dW, db = calculate_grads(Y, A, W, b)
        W, dW_prev = update_variables_RMSProp(0.001, 0.9, 1e-8, W, dW, dW_prev)
        b, db_prev = update_variables_RMSProp(0.001, 0.9, 1e-8, b, db, db_prev)
    A = forward_prop(X, W, b)
    cost = calculate_cost(Y, A)
    print('Cost after {} iterations: {}'.format(1000, cost))
```

```
Cost after 0 iterations: 4.365105010037203
Cost after 100 iterations: 1.3708321848806053
Cost after 200 iterations: 0.22693392990308764
Cost after 300 iterations: 0.05133394800221906
Cost after 400 iterations: 0.01836557116372359
Cost after 500 iterations: 0.008176390663315372
Cost after 600 iterations: 0.004091348850058557
Cost after 700 iterations: 0.002195647208708407
Cost after 800 iterations: 0.001148167933229118
Cost after 900 iterations: 0.0005599361043400206
Cost after 1000 iterations: 0.0002655839831275339
```

### [create_RMSProp_op](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/8-RMSProp.py "create_RMSProp_op")
Creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm.
``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
create_RMSProp_op = __import__('8-RMSProp').create_RMSProp_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        train_op = create_RMSProp_op(loss, 0.001, 0.9, 1e-8)
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            if not (i % 100):
                cost = sess.run(loss, feed_dict={x:X, y:Y_oh})
                print('Cost after {} iterations: {}'.format(i, cost))
            sess.run(train_op, feed_dict={x:X, y:Y_oh})
        cost, Y_pred_oh = sess.run((loss, y_pred), feed_dict={x:X, y:Y_oh})
        print('Cost after {} iterations: {}'.format(1000, cost))
```

```
Cost after 0 iterations: 2.8232274055480957
Cost after 100 iterations: 0.48531609773635864
Cost after 200 iterations: 0.21557031571865082
Cost after 300 iterations: 0.13388566672801971
Cost after 400 iterations: 0.07422538101673126
Cost after 500 iterations: 0.05024252086877823
Cost after 600 iterations: 0.02709660679101944
Cost after 700 iterations: 0.015626247972249985
Cost after 800 iterations: 0.008653616532683372
Cost after 900 iterations: 0.005407326854765415
Cost after 1000 iterations: 0.003452717326581478
```

### [update_variables_Adam](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/9-Adam.py "update_variables_Adam")
Updates a variable in place using the Adam optimization algorithm.
``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
update_variables_Adam = __import__('9-Adam').update_variables_Adam

def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m

    return cost

if __name__ == '__main__':
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y'].T
    X = X_3D.reshape((X_3D.shape[0], -1))

    nx = X.shape[1]
    np.random.seed(0)
    W = np.random.randn(nx, 1)
    b = 0
    dW_prev1 = np.zeros((nx, 1))
    db_prev1 = 0
    dW_prev2 = np.zeros((nx, 1))
    db_prev2 = 0
    for i in range(1000):
        A = forward_prop(X, W, b)
        if not (i % 100):
            cost = calculate_cost(Y, A)
            print('Cost after {} iterations: {}'.format(i, cost))
        dW, db = calculate_grads(Y, A, W, b)
        W, dW_prev1, dW_prev2 = update_variables_Adam(0.001, 0.9, 0.99, 1e-8, W, dW, dW_prev1, dW_prev2, i + 1)
        b, db_prev1, db_prev2 = update_variables_Adam(0.001, 0.9, 0.99, 1e-8, b, db, db_prev1, db_prev2, i + 1)
    A = forward_prop(X, W, b)
    cost = calculate_cost(Y, A)
    print('Cost after {} iterations: {}'.format(1000, cost))
```

```
Cost after 0 iterations: 4.365105010037203
Cost after 100 iterations: 1.5950468370180395
Cost after 200 iterations: 0.390276184856453
Cost after 300 iterations: 0.13737908627614337
Cost after 400 iterations: 0.06963385247882238
Cost after 500 iterations: 0.043186805401891
Cost after 600 iterations: 0.029615890163981955
Cost after 700 iterations: 0.02135952185721115
Cost after 800 iterations: 0.01576513402620876
Cost after 900 iterations: 0.011813533123333355
Cost after 1000 iterations: 0.008996494409788116
```

### [create_Adam_op](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/10-Adam.py "create_Adam_op")
Creates the training operation for a neural network in tensorflow using the Adam optimization algorithm.
``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
create_Adam_op = __import__('10-Adam').create_Adam_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        train_op = create_Adam_op(loss, 0.001, 0.9, 0.99, 1e-8)
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            if not (i % 100):
                cost = sess.run(loss, feed_dict={x:X, y:Y_oh})
                print('Cost after {} iterations: {}'.format(i, cost))
            sess.run(train_op, feed_dict={x:X, y:Y_oh})
        cost, Y_pred_oh = sess.run((loss, y_pred), feed_dict={x:X, y:Y_oh})
        print('Cost after {} iterations: {}'.format(1000, cost))
```

```
Cost after 0 iterations: 2.8232274055480957
Cost after 100 iterations: 0.17724855244159698
Cost after 200 iterations: 0.0870152935385704
Cost after 300 iterations: 0.03907731547951698
Cost after 400 iterations: 0.014239841140806675
Cost after 500 iterations: 0.0048021236434578896
Cost after 600 iterations: 0.0018489329377189279
Cost after 700 iterations: 0.000814757077023387
Cost after 800 iterations: 0.00038969298475421965
Cost after 900 iterations: 0.00019614089978858829
Cost after 1000 iterations: 0.00010206626757280901
```

### [learning_rate_decay](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/11-learning_rate_decay.py "learning_rate_decay")
Updates the learning rate using inverse time decay in numpy.
``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
learning_rate_decay = __import__('11-learning_rate_decay').learning_rate_decay

if __name__ == '__main__':
    alpha_init = 0.1
    for i in range(100):
        alpha = learning_rate_decay(alpha_init, 1, i, 10)
        print(alpha)
```

### [learning_rate_decay](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/12-learning_rate_decay.py "learning_rate_decay")
Creates a learning rate decay operation in tensorflow using inverse time decay.
``` python
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        global_step = tf.Variable(0, trainable=False)
        alpha = 0.1
        alpha = learning_rate_decay(alpha, 1, global_step, 10)
        train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss, global_step=global_step)
        init = tf.global_variables_initializer()
        sess.run(init)       
        for i in range(100):
            a = sess.run(alpha)
            print(a)
            sess.run(train_op, feed_dict={x:X, y:Y_oh})
```

### [batch_norm](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/13-batch_norm.py "batch_norm")
Normalizes an unactivated output of a neural network using batch normalization.
``` python
#!/usr/bin/env python3

import numpy as np
batch_norm = __import__('13-batch_norm').batch_norm

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    Z = np.concatenate((a, b, c), axis=1)
    gamma = np.random.rand(1, 3)
    beta = np.random.rand(1, 3)
    print(Z[:10])
    Z_norm = batch_norm(Z, gamma, beta, 1e-8)
    print(Z_norm[:10])
```

### [create_batch_norm_layer](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/14-batch_norm.py "create_batch_norm_layer")
Creates a batch normalization layer for a neural network in tensorflow.
``` python
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    X = X_3D.reshape((X_3D.shape[0], -1))

    tf.set_random_seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    a = create_batch_norm_layer(x, 256, tf.nn.tanh)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(a, feed_dict={x:X[:5]}))
```

### [model](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-optimization/15-model.py "model")
Builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization.
``` python
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
model = __import__('15-model').model

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]

    np.random.seed(0)
    tf.set_random_seed(0)
    save_path = model((X_train, Y_train_oh), (X_valid, Y_valid_oh), layer_sizes,
                                 activations, save_path='./model.ckpt')
    print('Model saved in path: {}'.format(save_path))
```

```
After 0 epochs:
    Training Cost: 2.5810317993164062
    Training Accuracy: 0.16808000206947327
    Validation Cost: 2.5596187114715576
    Validation Accuracy: 0.16859999299049377
    Step 100:
        Cost: 0.297500342130661
        Accuracy 0.90625
    Step 200:
        Cost: 0.27544915676116943
        Accuracy 0.875

    ...

    Step 1500:
        Cost: 0.09414251148700714
        Accuracy 1.0
After 1 epochs:
    Training Cost: 0.13064345717430115
    Training Accuracy: 0.9625800251960754
    Validation Cost: 0.14304184913635254
    Validation Accuracy: 0.9595000147819519

...

After 4 epochs:
    Training Cost: 0.03584253042936325
    Training Accuracy: 0.9912999868392944
    Validation Cost: 0.0853486955165863
    Validation Accuracy: 0.9750999808311462
    Step 100:
        Cost: 0.03150765225291252
        Accuracy 1.0
    Step 200:
        Cost: 0.020879564806818962
        Accuracy 1.0

    ...

    Step 1500:
        Cost: 0.015160675160586834
        Accuracy 1.0
After 5 epochs:
    Training Cost: 0.025094907730817795
    Training Accuracy: 0.9940199851989746
    Validation Cost: 0.08191727101802826
    Validation Accuracy: 0.9750999808311462
Model saved in path: ./model.ckpt
```