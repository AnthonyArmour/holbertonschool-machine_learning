# Q-Learning project

# Tasks

### [Load Env](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x00-q_learning/0-load_env.py "Load Env")
Loads "FrozenLake" Open AI gym environment.

``` python
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
import numpy as np

np.random.seed(0)
env = load_frozen_lake()
print(env.desc)
print(env.P[0][0])
env = load_frozen_lake(is_slippery=True)
print(env.desc)
print(env.P[0][0])
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
print(env.desc)
env = load_frozen_lake(map_name='4x4')
print(env.desc)
```
---

### [Q Init](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x00-q_learning/1-q_init.py "Q Init")
Initializes Q-table for Q-learning.

``` python
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init

env = load_frozen_lake()
Q = q_init(env)
print(Q.shape)
env = load_frozen_lake(is_slippery=True)
Q = q_init(env)
print(Q.shape)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)
print(Q.shape)
env = load_frozen_lake(map_name='4x4')
Q = q_init(env)
print(Q.shape)
```
---

### [Epsilon Greedy](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x00-q_learning/2-epsilon_greedy.py "Epsilon Greedy")
Function for guiding exploration and exploitation of Q-learning.

``` python
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
import numpy as np

desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)
Q[7] = np.array([0.5, 0.7, 1, -1])
np.random.seed(0)
print(epsilon_greedy(Q, 7, 0.5))
np.random.seed(1)
print(epsilon_greedy(Q, 7, 0.5))
```
---

### [Q Learning](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x00-q_learning/3-q_learning.py "Q Learning")
Function for training a Q-learning agent to play "FrozenLake".

``` python
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
import numpy as np

np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, total_rewards  = train(env, Q)
print(Q)
split_rewards = np.split(np.array(total_rewards), 10)
for i, rewards in enumerate(split_rewards):
    print((i+1) * 500, ':', np.mean(rewards))
```
---

### [Play](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x00-q_learning/4-play.py "Play")
Function that shows a Q-learning agent play "FrozenLake".

``` python
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
play = __import__('4-play').play

import numpy as np

np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, total_rewards  = train(env, Q)
print(play(env, Q))
```
---