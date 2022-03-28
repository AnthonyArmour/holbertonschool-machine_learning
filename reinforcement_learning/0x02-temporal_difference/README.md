[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Monte Carlo, SARSA Lambda, and TD Lambda Reinforcement Learning Algorithms

Monte Carlo methods are a broad class of computational algorithms that rely on repeated random sampling to obtain numerical results. The underlying concept is to use randomness to solve problems that might be deterministic in principle. In principle, Monte Carlo methods can be used to solve any problem having a probabilistic interpretation. By the law of large numbers, integrals described by the expected value of some random variable can be approximated by taking the empirical mean of independent samples of the variable. The computational cost of a Monte Carlo simulation can be staggeringly high.


Temporal difference (TD) learning is a class of model-free reinforcement learning methods which learn by bootstrapping from the current estimate of the value function. These methods sample from the environment and perform updates based on current estimates.

While Monte Carlo methods only adjust their estimates once the final outcome is known, TD methods adjust predictions to match later, more accurate, predictions about the future before the final outcome is known.

State–action–reward–state–action (SARSA) is an on-policy algorithm for learning a Markov decision process policy. This name simply reflects the fact that the main function for updating the Q-value depends on the current state of the agent "S1", the action the agent chooses "A1", the reward "R" the agent gets for choosing this action, the state "S2" that the agent enters after taking that action, and finally the next action "A2" the agent chooses in its new state.

## References

linked lectures

---

[![Model-Free-Prediction](https://img.youtube.com/vi/PnHCvfgC_ZA&t=2285s/0.jpg)](https://www.youtube.com/watch?v=PnHCvfgC_ZA&t=2285s)

[![Model-Free-Control](https://img.youtube.com/vi/0g4j2k_Ggc4/0.jpg)](https://www.youtube.com/watch?v=0g4j2k_Ggc4)

---


## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| gym                | ^0.20.0 |


## Tasks

### [Monte Carlo]()
Trains a value table using the mote carlo learning algorithm.

``` python
#!/usr/bin/env python3

import gym
import numpy as np
monte_carlo = __import__('0-monte_carlo').monte_carlo

np.random.seed(0)

env = gym.make('FrozenLake8x8-v0')
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

def policy(s):
    p = np.random.uniform()
    if p > 0.5:
        if s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
            return RIGHT
        elif s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
            return DOWN
        elif s // 8 != 0 and env.desc[s // 8 - 1, s % 8] != b'H':
            return UP
        else:
            return LEFT
    else:
        if s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
            return DOWN
        elif s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
            return RIGHT
        elif s % 8 != 0 and env.desc[s // 8, s % 8 - 1] != b'H':
            return LEFT
        else:
            return UP

V = np.where(env.desc == b'H', -1, 1).reshape(64).astype('float64') 
np.set_printoptions(precision=2)
env.seed(0)
V = monte_carlo(env, V, policy)
```

---

### [TD Lambda]()
Trains a value table using the TD lambda learning algorithm.

``` python
#!/usr/bin/env python3

import gym
import numpy as np
td_lambtha = __import__('1-td_lambtha').td_lambtha

np.random.seed(0)

env = gym.make('FrozenLake8x8-v0')
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

def policy(s):
    p = np.random.uniform()
    if p > 0.5:
        if s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
            return RIGHT
        elif s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
            return DOWN
        elif s // 8 != 0 and env.desc[s // 8 - 1, s % 8] != b'H':
            return UP
        else:
            return LEFT
    else:
        if s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
            return DOWN
        elif s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
            return RIGHT
        elif s % 8 != 0 and env.desc[s // 8, s % 8 - 1] != b'H':
            return LEFT
        else:
            return UP

V = np.where(env.desc == b'H', -1, 1).reshape(64).astype('float64') 
np.set_printoptions(precision=4)
V = td_lambtha(env, V, policy, 0.9)
```

---

### [SARSA]()
Trains a Q table using the SARSA learning algorithm.

``` python
#!/usr/bin/env python3

import gym
import numpy as np
sarsa_lambtha = __import__('2-sarsa_lambtha').sarsa_lambtha

np.random.seed(0)
env = gym.make('FrozenLake8x8-v0')
Q = np.random.uniform(size=(64, 4))
np.set_printoptions(precision=4)
Q = sarsa_lambtha(env, Q, 0.9)
```

