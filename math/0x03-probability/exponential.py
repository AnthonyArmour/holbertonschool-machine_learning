#!/usr/bin/env python3
"""Module for Exponential Probability Distribution"""


class Exponential():
    """Exponential Probability Distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Exponential init"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        elif type(data) is not list:
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")
        else:
            self.lambtha = 1 / float(sum(data) / len(data))

    def pdf(self, x):
        """Exponential Probability Distribution Function"""
        e = 2.7182818285
        if x < 0:
            return 0
        return self.lambtha * (e ** (-1 * self.lambtha * x))

    def cdf(self, x):
        """Exponential Cumulative Distribution Function"""
        if x < 0:
            return 0
        return 1 - (self.pdf(x) / self.lambtha)
