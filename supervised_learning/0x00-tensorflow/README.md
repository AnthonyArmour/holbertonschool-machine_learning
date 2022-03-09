[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# My First TensorFlow Project
Making and training neural network for classification using tensorflow.

## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| tensorflow         | ^1.12   |

### [Train](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-tensorflow/6-train.py "Train")
Trains a classifier feed forward neural network using tensorflow.

``` python
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
train = __import__('6-train').train

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

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
    iterations = 1000

    tf.set_random_seed(0)
    save_path = train(X_train, Y_train_oh, X_valid, Y_valid_oh, layer_sizes,
                      activations, alpha, iterations, save_path="./model.ckpt")
    print("Model saved in path: {}".format(save_path))
```

```
After 0 iterations:
    Training Cost: 2.8232274055480957
    Training Accuracy: 0.08726000040769577
    Validation Cost: 2.810533285140991
    Validation Accuracy: 0.08640000224113464
After 100 iterations:
    Training Cost: 0.8393384218215942
    Training Accuracy: 0.7824000120162964
    Validation Cost: 0.7826032042503357
    Validation Accuracy: 0.8061000108718872
After 200 iterations:
    Training Cost: 0.6094841361045837
    Training Accuracy: 0.8396000266075134
    Validation Cost: 0.5562412142753601
    Validation Accuracy: 0.8597999811172485

...

After 1000 iterations:
    Training Cost: 0.352960467338562
    Training Accuracy: 0.9004999995231628
    Validation Cost: 0.32148978114128113
    Validation Accuracy: 0.909600019454956
Model saved in path: ./model.ckpt
```
---

### [Evaluate](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-tensorflow/7-evaluate.py "Evaluate")
Evaluates our trained classifier neural network.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
evaluate = __import__('7-evaluate').evaluate

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_test_3D = lib['X_test']
    Y_test = lib['Y_test']
    X_test = X_test_3D.reshape((X_test_3D.shape[0], -1))
    Y_test_oh = one_hot(Y_test, 10)

    Y_pred_oh, accuracy, cost = evaluate(X_test, Y_test_oh, './model.ckpt')
    print("Test Accuracy:", accuracy)
    print("Test Cost:", cost)

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_test_3D[i])
        plt.title(str(Y_test[i]) + ' : ' + str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
```

```
Test Accuracy: 0.9096
Test Cost: 0.32148978
```

![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-tensorflow/images/tf-evaluation.png)