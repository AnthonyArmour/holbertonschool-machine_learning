#!/usr/bin/env python3
"""Binomial Probability Distribution"""


class Binomial():
    """Binomial Probability Distribution Class"""
    def __init__(self, data=None, n=1, p=0.5):
        """Binomial init"""
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            nns = []
            mean = sum(data) / len(data)
            v = sum([(i - mean) ** 2 for i in data]) / len(data)
            q = v / mean
            p = round(1 - q, 3)
            self.n = round(mean / p)
            self.p = round(mean / self.n, 3)

    def factorial(self, lst):
        """finds factorial"""
        fact, facts = 1, []
        for k in lst:
            for x in range(1, k + 1):
                fact = fact * x
            facts.append(int(fact))
            fact = 1
        return tuple(facts)

    def pmf(self, k):
        """Binomial Probability Mass Function"""
        if type(k) is not int:
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        n, x, n_x = self.factorial([self.n, k, self.n-k])
        choose = n / (x * n_x)
        P = (self.p**k) * ((1-self.p)**(self.n-k))
        return choose * P

    def cdf(self, k):
        """Binomial Cumulative Distribution Function"""
        if type(k) is not int:
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        return sum([self.pmf(i) for i in range(k + 1)])
