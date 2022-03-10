[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Introduction to Keras

## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| tensorflow         | ^2.6.0  |
| keras              | ^2.6.0  |

## Tasks:

### [build_model](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-keras/1-input.py "build_model")
Builds a neural network with the Keras library.
``` python
#!/usr/bin/env python3

build_model = __import__('1-input').build_model

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)
```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[<tf.Tensor 'dense/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_2/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_1/kernel/Regularizer/add:0' shape=() dtype=float32>]
```

### [optimize_model](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-keras/2-optimize.py "optimize_model")
Sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics.
``` python
#!/usr/bin/env python3

import tensorflow as tf

build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model

if __name__ == '__main__':
    model = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    optimize_model(model, 0.01, 0.99, 0.9)
    print(model.loss)
    print(model.metrics)
    opt = model.optimizer
    print(opt.__class__)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run((opt.lr, opt.beta_1, opt.beta_2))) 
```

```
categorical_crossentropy
['accuracy']
<class 'tensorflow.python.keras.optimizers.Adam'>
(0.01, 0.99, 0.9)
```

### [one_hot](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-keras/3-one_hot.py "one_hot")
One hot encode function to be used to reshape Y_label vector.
``` python
#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot

if __name__ == '__main__':
    labels = np.load('../data/MNIST.npz')['Y_train'][:10]
    print(labels)
    print(one_hot(labels)) 
```

```
[5 0 4 1 9 2 1 3 1 4]
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
```

### [train_model](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-keras/8-train.py "train_model")
Trains a model using mini-batch gradient descent.
``` python
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model 


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 1000
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=3, learning_rate_decay=True, alpha=alpha,
                save_best=True, filepath='network1.h5')
```

```
Train on 50000 samples, validate on 10000 samples

Epoch 00001: LearningRateScheduler reducing learning rate to 0.001.
Epoch 1/1000
50000/50000 [==============================] - 8s 157us/step - loss: 0.3508 - acc: 0.9202 - val_loss: 0.2174 - val_acc: 0.9602

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0005.
Epoch 2/1000
50000/50000 [==============================] - 8s 157us/step - loss: 0.1823 - acc: 0.9705 - val_loss: 0.1691 - val_acc: 0.9743

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0003333333333333333.
Epoch 3/1000
50000/50000 [==============================] - 6s 127us/step - loss: 0.1481 - acc: 0.9795 - val_loss: 0.1563 - val_acc: 0.9769

...

Epoch 00064: LearningRateScheduler reducing learning rate to 1.5625e-05.
Epoch 64/1000
50000/50000 [==============================] - 7s 133us/step - loss: 0.0517 - acc: 0.9990 - val_loss: 0.1029 - val_acc: 0.9827

Epoch 00065: LearningRateScheduler reducing learning rate to 1.5384615384615384e-05.
Epoch 65/1000
50000/50000 [==============================] - 5s 109us/step - loss: 0.0515 - acc: 0.9992 - val_loss: 0.1033 - val_acc: 0.9829

Epoch 00066: LearningRateScheduler reducing learning rate to 1.5151515151515151e-05.
Epoch 66/1000
50000/50000 [==============================] - 6s 112us/step - loss: 0.0510 - acc: 0.9993 - val_loss: 0.1034 - val_acc: 0.9830

Epoch 00067: LearningRateScheduler reducing learning rate to 1.4925373134328359e-05.
Epoch 67/1000
50000/50000 [==============================] - 6s 114us/step - loss: 0.0508 - acc: 0.9992 - val_loss: 0.1033 - val_acc: 0.9825
```

### [save_model](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-keras/9-model.py "save_model")
Saves or loads a Model.
``` python
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model 
model = __import__('9-model')

if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    network = model.load_model('network1.h5')
    batch_size = 32
    epochs = 1000
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=2, learning_rate_decay=True, alpha=0.001)
    model.save_model(network, 'network2.h5')
    network.summary()
    del network

    network2 = model.load_model('network2.h5')
    network2.summary()
```


### [weights](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-keras/10-weights.py "weights")
Saves and loads weights of a model.
``` python
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
weights = __import__('10-weights')

if __name__ == '__main__':

    network = model.load_model('network2.h5')
    weights.save_weights(network, 'weights2.h5')
    del network

    network2 = model.load_model('network1.h5')
    print(network2.get_weights())
    weights.load_weights(network2, 'weights2.h5')
    print(network2.get_weights())
```


### [config](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-keras/11-config.py "config")
Saves config of model to json and loads model with specific config from json.
``` python
#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

# Imports
model = __import__('9-model')
config = __import__('11-config')

if __name__ == '__main__':
    network = model.load_model('network1.h5')
    config.save_config(network, 'config1.json')
    del network

    network2 = config.load_config('config1.json')
    network2.summary()
    print(network2.get_weights())
```


### [test_model](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-keras/12-test.py "test_model")
Tests model on testing data.
``` python
#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
test_model = __import__('12-test').test_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']
    Y_test_oh = one_hot(Y_test)

    network = load_model('network2.h5')
    print(test_model(network, X_test, Y_test_oh))
```

```
10000/10000 [==============================] - 0s 43us/step
[0.09121923210024833, 0.9832]
```

### [predict](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x02-keras/13-predict.py "predict")
Makes a prediction using a neural network.
``` python
#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
predict = __import__('13-predict').predict


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']

    network = load_model('network2.h5')
    Y_pred = predict(network, X_test)
    print(Y_pred)
    print(np.argmax(Y_pred, axis=1))
    print(Y_test)
```

```
[[1.09882777e-07 1.85020565e-06 7.01209501e-07 ... 9.99942422e-01
  2.60075751e-07 8.19494835e-06]
 [1.37503928e-08 1.84829651e-06 9.99997258e-01 ... 2.15385221e-09
  8.63893135e-09 8.08128995e-14]
 [1.03242555e-05 9.99097943e-01 1.67965060e-04 ... 5.23889903e-04
  7.54134162e-05 1.10524084e-07]
 ...
 [1.88145090e-11 5.88180065e-08 1.43965796e-12 ... 3.95040814e-07
  1.28503856e-08 2.26610467e-07]
 [2.37400890e-08 2.48911092e-09 1.20860308e-10 ... 1.69956849e-08
  5.97703838e-05 3.89016153e-10]
 [2.68221925e-08 1.28844213e-10 5.13091347e-09 ... 1.14895975e-11
  1.83396942e-09 7.46730282e-12]]
[7 2 1 ... 4 5 6]
[7 2 1 ... 4 5 6]
```