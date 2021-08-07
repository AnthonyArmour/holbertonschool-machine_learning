#!/usr/bin/env python3
"""Deep Neural Network Module"""


import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """
       Deep Neural Network Class, used for binary
       classification on handwritten digits
    """

    def __init__(self, nx, layers, activation='sig'):
        """init method for DeepNeuralNetwork class"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if False in (np.array(layers) > 0):
            raise TypeError("layers must be a list of positive integers")
        if activation != "tanh" or activation != "sig":
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        layers = [nx] + layers
        for x, l in enumerate(layers[1:], start=1):
            self.__weights["W{}".format(x)] = (
                np.random.randn(l, layers[x - 1]) *
                np.sqrt(2/(layers[x - 1]))
            )
            self.__weights["b{}".format(x)] = np.zeros((l, 1))

    @property
    def activation(self):
        """getter for activation function attribute"""
        return self.__activation

    @property
    def cache(self):
        """getter for cache dictionary"""
        return self.__cache

    @property
    def L(self):
        """getter for num of layers, L"""
        return self.__L

    @property
    def weights(self):
        """getter for weights dictionary"""
        return self.__weights

    def save(self, filename):
        """Saves pickled object to .pkl file"""
        if filename.endswith(".pkl") is False:
            filename += ".pkl"
        with open(filename, "wb") as fh:
            pickle.dump(self, fh)

    def load(filename):
        """Loads object from pickle file"""
        try:
            with open(filename, "rb") as fh:
                obj = pickle.load(fh)
            return obj
        except Exception:
            return None

    def sigNN(self, layer):
        """sigmoid method for hidden layers"""
        Z = (
            np.matmul(self.__weights["W{}".format(layer)],
                        self.__cache["A{}".format(layer - 1)]) +
            self.__weights["b{}".format(layer)]
            )
        self.__cache["A{}".format(layer)] = 1/(1 + np.exp(-Z))

    def tanhNN(self, layer):
        """Tanh method for hidden layers"""
        Z = (
            np.matmul(self.__weights["W{}".format(layer)],
                        self.__cache["A{}".format(layer - 1)]) +
            self.__weights["b{}".format(layer)]
            )
        A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
        self.__cache["A{}".format(layer)] = A


    # def dtanh(self, layer, cache):
    #     """derivative of tanh method for NN"""
    #     return 1 - (self.__cache["A{}".format(layer - 1)] ** 2)

    def dsig_tanh(self, layer, cache):
        """derivative of sig for NN"""
        if self.__activation == "sig":
            return self.cache["A{}".format(layer)] * (1 - cache["A{}".format(layer)])
        else:
            return 1 - (self.__cache["A{}".format(layer)] ** 2)


    def forward_prop(self, X):
        """
           Forward Propagation method for
           Deep Neural Network using sigmoid
           activation function
        """
        # print("very start of forward prop")
        self.__cache["A0"] = X
        for layer in range(1, self.__L):
            if self.__activation == "sig":
                self.sigNN(layer)
            else:
                self.tanhNN(layer)
            # print("before iteration")
            # print(self.__weights["W{}".format(layer)].shape, "\n\n\n")
            # print(self.__cache["A{}".format(layer - 1)].shape)
        Z = (
            np.matmul(self.__weights["W{}".format(self.__L)],
                        self.__cache["A{}".format(self.__L - 1)]) +
            self.__weights["b{}".format(self.__L)]
            )
        T = np.exp(Z)
        # print("sum of t axis is 0:", np.sum(T, axis = 0))
        # print("length of L:", self.__L)
        self.__cache["A{}".format(self.__L)] = T/np.sum(T, axis = 0)
        #     print("after iteration")
        # print("end of forward prop")
        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Logistic Regression Cost Function"""
        mth = -1/A.shape[1]
        # costs = (Y * np.log(A)) + ((1.0000001 - Y) * np.log(1.0000001 - A))
        costs = Y * np.log(A)
        return np.sum(costs) * mth

    def evaluate(self, X, Y):
        """Evaluates the predictions made and the cost"""
        predictions, cache = self.forward_prop(X)
        cost = self.cost(Y, predictions)
        # print("prediction shape: ", predictions.shape)
        for x, max in enumerate(np.amax(predictions, axis=0)):
            predictions.T[x] = predictions.T[x] == max
        evaluation = predictions.astype(int)
        # eval_bool = predictions >= 0.5
        # evaluation = eval_bool.astype(int)
        return evaluation, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        mth = 1/cache["A1"].shape[1]
        partials = {}
        new_weights = {}
        partials["Z{}".format(self.__L)] = cache["A{}".format(self.__L)] - Y
        partials["W{}".format(self.__L)] = (
            mth * np.matmul(partials["Z{}".format(self.__L)],
                            cache["A{}".format(self.__L - 1)].T)
            )
        new_weights["W{}".format(self.__L)] = (
            self.__weights["W{}".format(self.__L)] -
            (alpha * partials["W{}".format(self.__L)])
        )
        partials["b{}".format(self.__L)] = (
            mth * np.sum(partials["Z{}".format(self.__L)],
                         axis=1, keepdims=True)
        )
        new_weights["b{}".format(self.__L)] = (
            self.__weights["b{}".format(self.__L)] -
            (alpha * partials["b{}".format(self.__L)])
        )

        for layer in range(self.__L - 1, 0, -1):
            partials["Z{}".format(layer)] = (
                np.matmul(self.__weights["W{}".format(layer + 1)].T,
                          partials["Z{}".format(layer + 1)]) *
                self.dsig_tanh(layer, cache) #activation derivative
            )
            partials["W{}".format(layer)] = (
                mth * np.matmul(partials["Z{}".format(layer)],
                                cache["A{}".format(layer - 1)].T)
            )
            new_weights["W{}".format(layer)] = (
                self.__weights["W{}".format(layer)] -
                (alpha * partials["W{}".format(layer)])
            )
            partials["b{}".format(layer)] = (
                mth * np.sum(partials["Z{}".format(layer)],
                             axis=1, keepdims=True)
            )
            new_weights["b{}".format(layer)] = (
                self.__weights["b{}".format(layer)] -
                (alpha * partials["b{}".format(layer)])
            )
        self.__weights = new_weights

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """
           Method that uses forward porpagation
           and back propagation to train deep
           neural net
        """
        # print("very start of train method")
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            costs, x_points = [], []
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        # print("before for loop in train method")
        for x in range(iterations):
            AL, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha=alpha)
            if ((x == 0 or x % step == 0) and
               (verbose is True or graph is True)):
                cost = self.cost(Y, self.__cache["A{}".format(self.__L)])
                costs.append(cost), x_points.append(x)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(x, cost))
        if verbose is True or graph is True:
            cost = self.cost(Y, self.__cache["A{}".format(self.__L)])
            costs.append(cost), x_points.append(iterations)
            if verbose is True:
                print("Cost after {} iterations: {}".format(iterations, cost))
        if graph is True:
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.suptitle("Training Cost")
            plt.plot(x_points, costs, "b")
            plt.show()
        return self.evaluate(X, Y)
