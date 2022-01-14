# Recurrent Neural Networks

## Tasks

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