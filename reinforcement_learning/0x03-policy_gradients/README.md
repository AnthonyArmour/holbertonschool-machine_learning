[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Policy Gradients
Solving the CartPole environment using the Monte-Carlo Policy Gradient REINFORCE algorithm.

Policy gradient methods are a type of reinforcement learning technique that relies upon optimizing parametrized policies with respect to long-term cumulative return (reward) by gradient descent. They do not suffer from many of the problems of traditional reinforcement learning approaches such as the lack of guarantees of a value function, the intractability problem resulting from uncertain state information and the complexity arising from continuous states & actions. They are by definition on-policy methods. Indeed in most applications, there exist many local maxima.


---

## References

linked lecture

---

[![Model-Free-Prediction](https://img.youtube.com/vi/KHZVXao4qXs/0.jpg)](https://www.youtube.com/watch?v=KHZVXao4qXs&t=2932s)

---


## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| gym                | ^0.20.0 |
| matplotlib         | ^3.4.3  |

---

## Results


![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x03-policy_gradients/assets/reward.jpg)

![gif](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/reinforcement_learning/0x03-policy_gradients/assets/CartPole.gif)

---

## Code

``` python
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from train import PolicyGradient

env = gym.make('CartPole-v1')

# To save cartpole animation set save_res=True
PG = PolicyGradient(save_res=True)

fig = plt.figure(figsize=(4, 5))
plt.axis('off')

scores = PG.train(env, 10000)
env.close()

# This block is for saving cartpole animation
plt.close()
ani = animation.ArtistAnimation(fig=fig, artists=PG.render.ims, interval=20)
ani.save("./CartPole.gif", writer="pillow")

# This block plots episodic rewards
plt.plot(np.arange(len(scores)), scores)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("./reward.jpg")
plt.close()

```