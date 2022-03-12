[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Deep Q Learning
Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state. For any finite Markov decision process, Q-learning finds an optimal policy in the sense of maximizing the expected value of the total reward over any and all successive steps, starting from the current state. "Q" refers to the function that the algorithm computes – the expected rewards for an action taken in a given state.

In this project, I use the keras-rl framework for training a deep Q learning agent to approximate the Q value function for the game Atari Breakout.

## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |
| keras              | ^2.6.0  |
| keras-rl2          | ^1.0.5  |
| gym                | ^0.20.0 |

## Results
![gif](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x01-deep_q_learning/DQN_AtariBreakout/anim.gif)


### [Train model](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x01-deep_q_learning/train.py)

```
Run -> python3 train.py

... training status
```

### [Run model test and animation](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x01-deep_q_learning/play.py)

```
Run -> python3 play.py

You will be asked to save the animation as a gif.
(yes/no)
```


