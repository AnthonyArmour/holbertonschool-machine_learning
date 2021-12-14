# Project on Markov Chains and Hidden Markov Models

## Tasks

### [Markov Chain](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/0-markov_chain.py "Markov Chain")
Determines the probability of a markov chain being in a particular state after a specified number of iterations.
``` python
#!/usr/bin/env python3

import numpy as np
markov_chain = __import__('0-markov_chain').markov_chain

if __name__ == "__main__":
    P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3], [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
    s = np.array([[1, 0, 0, 0]])
    print(markov_chain(P, s, 300))
```
---

### [Regular Chains](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/1-regular.py "Regular Chains")
Determines the steady state probabilities of a regular markov chain.
``` python
#!/usr/bin/env python3

import numpy as np
regular = __import__('1-regular').regular

if __name__ == '__main__':
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    print(regular(a))
    print(regular(b))
    print(regular(c))
```
---

### [Absorbing Chains](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/2-absorbing.py "Absorbing Chains")
Determines if a markov chain is absorbing.
``` python
#!/usr/bin/env python3

import numpy as np
absorbing = __import__('2-absorbing').absorbing

if __name__ == '__main__':
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    d = np.array([[1, 0, 0, 0, 0],
                  [0.25, 0.75, 0, 0, 0],
                  [0, 0, 0.5, 0.2, 0.3],
                  [0, 0, 0.3, 0.5, .2],
                  [0, 0, 0.2, 0.3, 0.5]])
    print(absorbing(a))
    print(absorbing(b))
    print(absorbing(c))
```
---

### [The Forward Algorithm](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/3-forward.py "The Forward Algorithm")
Performs the forward algorithm for a hidden markov model.
``` python
#!/usr/bin/env python3

import numpy as np
forward = __import__('3-forward').forward

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    P, F = forward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(F)
```
---

### [The Viretbi Algorithm](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/4-viterbi.py "The Viretbi Algorithm")
Calculates the most likely sequence of hidden states for a hidden markov model.
``` python
#!/usr/bin/env python3

import numpy as np
viterbi = __import__('4-viterbi').viterbi

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    path, P = viterbi(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(path)
```
---

### [The Backward Algorithm](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/5-backward.py "The Backward Algorithm")
Performs the backward algorithm for a hidden markov model.
``` python
#!/usr/bin/env python3

import numpy as np
backward = __import__('5-backward').backward

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    P, B = backward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(B)
```
---

### [The Baum-Welch Algorithm]( "The Baum-Welch Algorithm")
Module contains function that performs the Baum-Welch algorithm for finding locally optimal transition and emission probabilities for a Hidden Markov Model.