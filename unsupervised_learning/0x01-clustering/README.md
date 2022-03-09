[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Clustering Project

## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |
| scipy              | ^1.7.3  |

## Tasks

### K-Means Algorithm
k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. k-means clustering minimizes within-cluster variances. The unsupervised k-means algorithm has a loose relationship to the k-nearest neighbor classifier, a popular supervised machine learning technique for classification that is often confused with k-means due to the name.

### [Kmeans](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/1-kmeans.py "Kmeans")
Performs K-means on a dataset.

``` python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
kmeans = __import__('1-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 5)
    print(C)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    plt.show()
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/k-means.png)

### [Optimize K](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/3-optimum.py "Optimize K")
Tests for the optimum number of clusters by variance.

``` python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
optimum_k = __import__('3-optimum').optimum_k

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    results, d_vars = optimum_k(X, kmax=10)
    plt.scatter(list(range(1, 11)), d_vars)
    plt.xlabel('Clusters')
    plt.ylabel('Delta Variance')
    plt.title('Optimizing K-means')
    plt.show()
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/optimal-k.png)
---

### Multivariate Gaussian Mixture Model
Formally a mixture model corresponds to the mixture distribution that represents the probability distribution of observations in the overall population. Density plots are used to analyze the density of high dimensional features. If multi-model densities are observed, then it is assumed that a finite set of densities are formed by a finite set of normal mixtures. A multivariate Gaussian mixture model is used to cluster the feature data into k number of groups where k represents each state of the machine. The machine state can be a normal state, power off state, or faulty state. Each formed cluster can be diagnosed using techniques such as spectral analysis.


### [Expectation](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/6-expectation.py "Expectation")
Calculates the expectation step in the EM algorithm for a GMM.


### [Maximization](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/7-maximization.py "Maximization")
Calculates the maximization step in the EM algorithm for a GMM.


### [Expectation Maximization](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/8-EM.py "Expectation Maximization")
Performs the expectation maximization for a GMM.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    k = 4
    pi, m, S, g, l = expectation_maximization(X, k, 150, verbose=True)
    clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
    plt.scatter(X[:, 0], X[:, 1], s=20, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, c=np.arange(k), marker='*')
    plt.show()
```

```
Log Likelihood after 0 iterations: -652797.78665
Log Likelihood after 10 iterations: -94855.45662
Log Likelihood after 20 iterations: -94714.52057
Log Likelihood after 30 iterations: -94590.87362
Log Likelihood after 40 iterations: -94440.40559
Log Likelihood after 50 iterations: -94439.93891
Log Likelihood after 52 iterations: -94439.93889
```

![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/expectation-maximization.png)
---

### [Bayesian Information Criterion](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/9-BIC.py "Bayesian Information Criterion")
Finds the best number of clusters for a GMM using the bayesian information criterion.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
BIC = __import__('9-BIC').BIC

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    best_k, best_result, l, b = BIC(X, kmin=1, kmax=10)
    # print(best_k)
    # print(best_result)
    # print(l)
    # print(b)
    x = np.arange(1, 11)
    plt.plot(x, l, 'r')
    plt.xlabel('Clusters')
    plt.ylabel('Log Likelihood')
    plt.tight_layout()
    plt.show()
    plt.plot(x, b, 'b')
    plt.xlabel('Clusters')
    plt.ylabel('BIC')
    plt.tight_layout()
    plt.show()
```

![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/bayes-info-criterion-1.png)
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/bayes-info-criterion-2.png)
---

### [K-Means Sklearn](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/10-kmeans.py "K-Means Sklearn")
K-means clustering using sklearn.

### [GMM Sklearn](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/11-gmm.py "GMM Sklearn")
Gaussian mixture model using sklearn.

---

### [Agglomerative](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/12-agglomerative.py "Agglomerative")
Performs agglomerative clustering on a dataset using sklearn.cluster.hierarchy.

``` python
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
agglomerative = __import__('12-agglomerative').agglomerative

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=100)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=100)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    clss = agglomerative(X, 100)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.show()
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/agg-hierarchy.png)
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/agg-cluster.png)
---