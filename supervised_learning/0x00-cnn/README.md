# Convolutional Neural Network Project
## Performing forward and backward prop on conv nets with numpy, as well as building conv nets in tensorflow and keras.

* conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1))

> Performs forward propagation over a convolutional layer of a neural network.

* pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):

> Performs forward propagation over a pooling layer of a neural network.

* conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):

> Performs back propagation over a convolutional layer of a neural network.

* pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):

> Performs back propagation over a pooling layer of a neural network.

* lenet5(x, y):

> Modified version of the LeNet-5 architecture using tensorflow.

* lenet5(X):

> Modified version of the LeNet-5 architecture using keras.