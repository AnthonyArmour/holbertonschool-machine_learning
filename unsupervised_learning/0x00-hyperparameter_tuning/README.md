[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Gaussian Processes and Bayesian Optimization

Used to model a function given a small amount of data points, Gaussian process is a stochastic process (a collection of random variables indexed by time or space), such that every finite collection of those random variables has a multivariate normal distribution. Gaussian processes can be seen as an infinite-dimensional generalization of multivariate normal distributions. Gaussian processes fit the data by getting the mean over distributions and define uncertainty along the model by the variance along those sampled distributions.

Bayesian optimization is a sequential design strategy for global optimization of black-box functions that does not assume any functional forms. It is usually employed to optimize expensive-to-evaluate functions. It takes a model like a gaussian process to define a posterior and works by finding the sample location that maximizes the expected improvement of the objective function.

![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x00-hyperparameter_tuning/Images/plot_posterior.jpg)

In this project, we create a Gaussian process from scratch and use it for Bayesian optimization of a blackbox function. The school project continues the class for each file, the complete classes can be found in 2-gp.py and 5-bayes_opt.py


## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |

## Tasks

### [Gaussian Process](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x00-hyperparameter_tuning/2-gp.py "Gaussian Process")
Class that represents a noiseless 1D Gaussian Process.

``` python
#!/usr/bin/env python3

GP = __import__('2-gp').GaussianProcess
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_new = np.random.uniform(-np.pi, 2*np.pi, 1)
    Y_new = f(X_new)
    gp.update(X_new, Y_new)
```
---

### [Bayesian Optimization](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x00-hyperparameter_tuning/5-bayes_opt.py "Bayesian Optimization")
Class for bayesian optimization.

``` python
#!/usr/bin/env python3

BO = __import__('5-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2)
    X_opt, Y_opt = bo.optimize(50)
    print('Optimal X:', X_opt)
    print('Optimal Y:', Y_opt)
```
---

### Plot Bayesian Optimization Aquisition Function
``` python
#!/usr/bin/env python3

BO = __import__('4-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2, xsi=0.05)
    X_next, EI = bo.acquisition()

    print(EI)
    print(X_next)

    plt.scatter(X_init.reshape(-1), Y_init.reshape(-1), color='g')
    plt.plot(bo.X_s.reshape(-1), EI.reshape(-1), color='r')
    plt.axvline(x=X_next)
    plt.show()
```

![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x00-hyperparameter_tuning/Images/BO_Aquisition.jpg)
---

### [Plot Gaussian Process](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x00-hyperparameter_tuning/Plot_Gaussian_Process.py "Plot Gaussian Process")
Generates the plot at the top of the readme.

| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |
| scipy              | ^1.7.3  |

---