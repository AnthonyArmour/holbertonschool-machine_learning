[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)


# Recurrent Neural Networks
Recurrent Neural Networks (RNNs) are state of the art algorithms for sequential data. They are a powerful machine learning technique that achieves an internal memory which makes them perfectly suited for solving problems involving sequential data. RNNs are a robust type of neural network and gain their power by including an internal memory of input history to help make predictions on future time steps. RNNs are used to model data that have temporal dynamics. To understand RNNs it is important to have knowledge of a normal feed-forward neural network.

The meaningful difference between RNNs and traditional feed-forward neural networks is the way that they channel information. In a feed-forward network, the information moves in a single direction: from the input layer, through the hidden layers, to the output layer. The information never touches the same node twice, moving straight through the network. Each prediction is deterministic with respect to the network’s inputs. This is due to it’s set internal parameters which directly influences the forward pass of operations that make up the final prediction. Feed-forward networks only consider the current input and have no memory of previous inputs; thus, having no notion of order in time. The state of a feed-forward network after training could be thought of as “memory” from training, but this has nothing to do with a learned representation of temporal dynamics. They simply can’t remember anything from the past except for their internal weights learned during training. Here’s a truth to get a better picture: every unique input passed through a feed-forward network has the same output every time. In contrast, RNNs, can make a different prediction at a given time step for the same input.

In an RNN the information cycles through a loop. When it makes a prediction, it considers the current input and also what it has observed from previously received inputs. Here is an image representing the architectural difference between a Recurrent Neural Network and a traditional feed-forward neural network.

![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-RNNs/images/RNN.jpeg)

To put it simply, an RNN adds the immediate past to the present. Thus, an RNN has two inputs: the recurrent past and the present input. Like feed-forward networks, RNNs assign a weight matrix to it’s layer inputs, but differ from traditional networks in their application of weights to previous inputs. Their optimization procedure (backpropagation through time) also sets them apart from traditional feed-forward networks.

To understand backpropagation through time it’s important to understand traditional forward-propagation and backpropagation. These concepts are a lot to dive into, so I will try to be as brief as possible.


Forward-propagation is the prediction step of a neural network. It is simply a series of linear and non-linear operations that works on an initial input. The weights associated with each node in the layers of the network parameterize each operation. By the end of a single forward pass a prediction is made, allowing for an error to be calculated with respect to the ground truth that the model should have predicted. The error represents how bad the network’s prediction turned out.

Backpropagation (Backprop) is a machine learning optimization algorithm used for training a neural network. It is used for calculating the gradients of an error function with respect to the network’s weights. The algorithm works it’s way backwards through the layers of the network to find the partial derivatives of the errors with respect to the network’s weights. These derivatives are then used to adjust the weights of the network to decrease prediction error when training on a dataset.

In a basic sense, training a neural network is an iterative algorithm that consists of two steps: first, using forward-propagation, given an input, to make a prediction and calculate an error. Second, performing backprop to adjust the internal parameters of the model with respect to the error, intended to improve the model’s performance in the next iteration.

Backpropagation through time (BPTT) is simply the backpropagation algorithm on an unrolled RNN, allowing the optimization algorithm to take into account the temporal dynamics in the architecture. The BPTT is a necessary adjustment since the error of a given time step depends on the previous time step. Within BPTT the error from the last time step is back propagated to the first time step, while unrolling all the time steps. It’s important to note that BPTT can be computationally expensive with a large number of time steps.

There are two big obstacles that RNNs need to deal with, but it is important to first understand what a gradient is. A gradient is the partial derivative of a function with respect to its inputs. To put it simply, a gradient measures the effect that a small change to a function’s inputs has on the function's output. You can also think of a gradient as the tangent line across a function, or the slope of a function at a given point. The two big problems with standard RNNs are exploding and vanishing gradients. Exploding gradients happen when large error gradients accumulate resulting in the model assigning progressively larger value updates to the model weights. This can often be solved with gradient clipping and gradient squashing. Vanishing gradients, the bigger problem of the two, happens when gradients of layers close to the output become small, increasing the speed at which the gradients of earlier layers will approach zero. This is because, as layers and/or steps increase, the product of gradients calculated in backprop becomes the product of values much smaller than one. Standard RNNs are extremely susceptible to the vanishing gradient problem, but, fortunately, additions like Long Short-Term Memory units (LSTMs) are well suited for solving this problem.


## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |

## Tasks
This project implements the forward pass of the following recurrent neural network architectures from scratch in numpy. There is nothing visually interesting about this project. More interesting implementations of RNNs using the keras API can be found in the root of this repository.

### [RNN Cell](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-RNNs/0-rnn_cell.py "RNN Cell")
Represents a cell of a simple RNN.

``` python
#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell

np.random.seed(0)
rnn_cell = RNNCell(10, 15, 5)
print("Wh:", rnn_cell.Wh)
print("Wy:", rnn_cell.Wy)
print("bh:", rnn_cell.bh)
print("by:", rnn_cell.by)
rnn_cell.bh = np.random.randn(1, 15)
rnn_cell.by = np.random.randn(1, 5)
h_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h, y = rnn_cell.forward(h_prev, x_t)
print(h.shape)
print(h)
print(y.shape)
print(y)
```
---

### [RNN](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-RNNs/1-rnn.py "RNN")
Performs forward propagation for a simple RNN.

``` python
#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
rnn = __import__('1-rnn').rnn

np.random.seed(1)
rnn_cell = RNNCell(10, 15, 5)
rnn_cell.bh = np.random.randn(1, 15)
rnn_cell.by = np.random.randn(1, 5)
X = np.random.randn(6, 8, 10)
h_0 = np.zeros((8, 15))
H, Y = rnn(rnn_cell, X, h_0)
print(H.shape)
print(H)
print(Y.shape)
print(Y)
```
---

### [GRU Cell](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-RNNs/2-gru_cell.py "GRU Cell")
Represents a gated recurrent unit for a RNN.

``` python
#!/usr/bin/env python3

import numpy as np
GRUCell = __import__('2-gru_cell').GRUCell

np.random.seed(2)
gru_cell = GRUCell(10, 15, 5)
print("Wz:", gru_cell.Wz)
print("Wr:", gru_cell.Wr)
print("Wh:", gru_cell.Wh)
print("Wy:", gru_cell.Wy)
print("bz:", gru_cell.bz)
print("br:", gru_cell.br)
print("bh:", gru_cell.bh)
print("by:", gru_cell.by)
gru_cell.bz = np.random.randn(1, 15)
gru_cell.br = np.random.randn(1, 15)
gru_cell.bh = np.random.randn(1, 15)
gru_cell.by = np.random.randn(1, 5)
h_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h, y = gru_cell.forward(h_prev, x_t)
print(h.shape)
print(h)
print(y.shape)
print(y)
```
---

### [LSTM Cell](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-RNNs/3-lstm_cell.py "LSTM Cell")
Represents an LSTM unit for a RNN.

``` python
#!/usr/bin/env python3

import numpy as np
LSTMCell = __import__('3-lstm_cell').LSTMCell

np.random.seed(3)
lstm_cell = LSTMCell(10, 15, 5)
print("Wf:", lstm_cell.Wf)
print("Wu:", lstm_cell.Wu)
print("Wc:", lstm_cell.Wc)
print("Wo:", lstm_cell.Wo)
print("Wy:", lstm_cell.Wy)
print("bf:", lstm_cell.bf)
print("bu:", lstm_cell.bu)
print("bc:", lstm_cell.bc)
print("bo:", lstm_cell.bo)
print("by:", lstm_cell.by)
lstm_cell.bf = np.random.randn(1, 15)
lstm_cell.bu = np.random.randn(1, 15)
lstm_cell.bc = np.random.randn(1, 15)
lstm_cell.bo = np.random.randn(1, 15)
lstm_cell.by = np.random.randn(1, 5)
h_prev = np.random.randn(8, 15)
c_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h, c, y = lstm_cell.forward(h_prev, c_prev, x_t)
print(h.shape)
print(h)
print(c.shape)
print(c)
print(y.shape)
print(y)
```
---

### [Deep RNN](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-RNNs/4-deep_rnn.py "Deep RNN")
Performs forward propagation for a deep RNN.

``` python
#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
deep_rnn = __import__('4-deep_rnn').deep_rnn

np.random.seed(1)
cell1 = RNNCell(10, 15, 1)
cell2 = RNNCell(15, 15, 1)
cell3 = RNNCell(15, 15, 5)
rnn_cells = [cell1, cell2, cell3]
for rnn_cell in rnn_cells:
    rnn_cell.bh = np.random.randn(1, 15)
cell3.by = np.random.randn(1, 5)
X = np.random.randn(6, 8, 10)
H_0 = np.zeros((3, 8, 15))
H, Y = deep_rnn(rnn_cells, X, H_0)
print(H.shape)
print(H)
print(Y.shape)
print(Y)
```
---

### [Bidirectional Cell Forward](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-RNNs/5-bi_forward.py "Bidirectional Cell Forward")
BidirectionalCell represents a bidirectional cell of an RNN - forward method calculates the hidden state in the backward direction for one time step.

``` python
#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('5-bi_forward'). BidirectionalCell

np.random.seed(5)
bi_cell =  BidirectionalCell(10, 15, 5)
print("Whf:", bi_cell.Whf)
print("Whb:", bi_cell.Whb)
print("Wy:", bi_cell.Wy)
print("bhf:", bi_cell.bhf)
print("bhb:", bi_cell.bhb)
print("by:", bi_cell.by)
bi_cell.bhf = np.random.randn(1, 15)
h_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h = bi_cell.forward(h_prev, x_t)
print(h.shape)
print(h)
```
---

### [Bidirectional Cell Backward](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-RNNs/6-bi_backward.py "Bidirectional Cell Backward")
BidirectionalCell represents a bidirectional cell of an RNN - backward method calculates the hidden state in the backward direction for one time step.

``` python
#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('6-bi_backward'). BidirectionalCell

np.random.seed(6)
bi_cell =  BidirectionalCell(10, 15, 5)
bi_cell.bhb = np.random.randn(1, 15)
h_next = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h = bi_cell.backward(h_next, x_t)
print(h.shape)
print(h)
```
---

### [Bidirectional Output](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-RNNs/7-bi_output.py "Bidirectional Output")
Calculates all outputs for the RNN.

``` python
#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('7-bi_output'). BidirectionalCell

np.random.seed(7)
bi_cell =  BidirectionalCell(10, 15, 5)
bi_cell.by = np.random.randn(1, 5)
H = np.random.randn(6, 8, 30)
Y = bi_cell.output(H)
print(Y.shape)
print(Y)
```
---

### [Bidirectional RNN](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-RNNs/8-bi_rnn.py "Bidirectional RNN")
Performs forward propagation for a bidirectional RNN.

``` python
#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('7-bi_output').BidirectionalCell
bi_rnn = __import__('8-bi_rnn').bi_rnn

np.random.seed(8)
bi_cell =  BidirectionalCell(10, 15, 5)
X = np.random.randn(6, 8, 10)
h_0 = np.zeros((8, 15))
h_T = np.zeros((8, 15))
H, Y = bi_rnn(bi_cell, X, h_0, h_T)
print(H.shape)
print(H)
print(Y.shape)
print(Y)
```
---