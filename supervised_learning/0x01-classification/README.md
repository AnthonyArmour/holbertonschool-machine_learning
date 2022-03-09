[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Neural Networks
A neural network is a network or circuit of  artificial neurons or nodes. The connections of biological neurons are modeled in artificial neural networks as weights between nodes. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. All inputs are modified by a weight and summed. This activity is referred to as a linear combination. Finally, an activation function controls the amplitude of the output. For example, an acceptable range of output is usually between 0 and 1, or it could be −1 and 1.

These artificial networks may be used for predictive modeling, adaptive control and applications where they can be trained via a dataset. Self-learning resulting from experience can occur within networks, which can derive conclusions from a complex and seemingly unrelated set of information.

## Classification
In machine learning, classification refers to a predictive modeling problem where a class label is predicted for a given example of input data. Examples of classification problems include: Given an example, classify if it is spam or not. Given a handwritten character, classify it as one of the known characters.


## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |


## Tasks
Making and training a classification neural network from scratch using numpy to classify hand written digits provided in the MNIST dataset. The assignment starts with building a single neuron for binary classification and builds up to a multi-layer perceptron network for classification of digits 0-9.


### [Class Neuron](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/7-neuron.py "Class Neuron")
Defines and trains a single neuron performing binary classification.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('7-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=3000)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

```
Cost after 0 iterations: 4.365104944262272
Cost after 100 iterations: 0.11955134491351888

...

Cost after 3000 iterations: 0.013386353289868338
```
---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/neuron-1.png)
---

```
Train cost: 0.013386353289868338
Train accuracy: 99.66837741808132%
Dev cost: 0.010803484515167197
Dev accuracy: 99.81087470449172%
```
---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/neuron-2.png)
---

### [Class NeuralNetwork](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/15-neural_network.py "Class NeuralNetwork")
Defines and trains a neural network with one hidden layer performing binary classification.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

NN = __import__('15-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X_train.shape[0], 3)
A, cost = nn.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

```
Cost after 0 iterations: 0.7917984405648547
Cost after 100 iterations: 0.4680930945144984

...

Cost after 5000 iterations: 0.024369225667283875
```

---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/NeuralNetwork-1.png)
---

```
Train cost: 0.024369225667283875
Train accuracy: 99.3999210422424%
Dev cost: 0.020330639788072768
Dev accuracy: 99.57446808510639%
```

---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/NeuralNetwork-2.png)
---


### [Class DeepNeuralNetwork](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/23-deep_neural_network.py "Class DeepNeuralNetwork")
Defines and trains a deep neural network performing binary classification.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('23-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X_train.shape[0], [5, 3, 1])
A, cost = deep.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = deep.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

```
Cost after 0 iterations: 0.6958649419170609
Cost after 100 iterations: 0.6444304786060048

...

Cost after 5000 iterations: 0.011671820326008168
```

---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/DeepNeuralNetwork-1.png)
---

```
Train cost: 0.011671820326008168
Train accuracy: 99.88945913936044%
Dev cost: 0.00924955213227925
Dev accuracy: 99.95271867612293%
```

---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/DeepNeuralNetwork-2.png)
---

### [Multiclass Classification](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/28-deep_neural_network.py "Multiclass Classification")
Updates the class DeepNeuralNetwork to perform multiclass classification.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep28 = __import__('28-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

lib= np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_test_3D = lib['X_test']
Y_test = lib['Y_test']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
X_test = X_test_3D.reshape((X_test_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)
Y_test_one_hot = one_hot_encode(Y_test, 10)

print('Sigmoid activation:')
deep28 = Deep28(X_train.shape[0], [5, 3, 1], activation='sig')
A_one_hot28, cost28 = deep28.train(X_train, Y_train_one_hot, iterations=100,
                                step=10, graph=False)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_train == A28) / Y_train.shape[0] * 100
print("Train cost:", cost28)
print("Train accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_valid, Y_valid_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_valid == A28) / Y_valid.shape[0] * 100
print("Validation cost:", cost28)
print("Validation accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_test, Y_test_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_test == A28) / Y_test.shape[0] * 100
print("Test cost:", cost28)
print("Test accuracy: {}%".format(accuracy28))
deep28.save('28-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A28[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.close()

print('\nTanh activaiton:')

deep28 = Deep28(X_train.shape[0], [5, 3, 1], activation='tanh')
A_one_hot28, cost28 = deep28.train(X_train, Y_train_one_hot, iterations=100,
                                step=10, graph=False)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_train == A28) / Y_train.shape[0] * 100
print("Train cost:", cost28)
print("Train accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_valid, Y_valid_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_valid == A28) / Y_valid.shape[0] * 100
print("Validation cost:", cost28)
print("Validation accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_test, Y_test_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_test == A28) / Y_test.shape[0] * 100
print("Test cost:", cost28)
print("Test accuracy: {}%".format(accuracy28))
deep28.save('28-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A28[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

```
Sigmoid activation:
Cost after 0 iterations: 0.4388904112857044
Cost after 10 iterations: 0.4377828804163359
Cost after 20 iterations: 0.43668839872612714
Cost after 30 iterations: 0.43560674736059446
Cost after 40 iterations: 0.43453771176806555
Cost after 50 iterations: 0.4334810815993252
Cost after 60 iterations: 0.43243665061046205
Cost after 70 iterations: 0.4314042165687683
Cost after 80 iterations: 0.4303835811615513
Cost after 90 iterations: 0.4293745499077264
Cost after 100 iterations: 0.42837693207206473
Train cost: 0.42837693207206456
Train accuracy: 88.442%
Validation cost: 0.39517557351173044
Validation accuracy: 89.64%
Test cost: 0.4074169894615401
Test accuracy: 89.0%
```

---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/multiclass-1.png)
---

```
Tanh activaiton:
Cost after 0 iterations: 0.1806181562229199
Cost after 10 iterations: 0.1801200954271858
Cost after 20 iterations: 0.1796242897834926
Cost after 30 iterations: 0.17913072860418564
Cost after 40 iterations: 0.1786394012066576
Cost after 50 iterations: 0.17815029691267442
Cost after 60 iterations: 0.1776634050478437
Cost after 70 iterations: 0.1771787149412177
Cost after 80 iterations: 0.1766962159250237
Cost after 90 iterations: 0.1762158973345138
Cost after 100 iterations: 0.1757377485079266
Train cost: 0.1757377485079266
Train accuracy: 95.006%
Validation cost: 0.17689309600397934
Validation accuracy: 95.13000000000001%
Test cost: 0.1809489808838737
Test accuracy: 94.77%
```

---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/multiclass-2.png)
---