#!/usr/bin/env python3
"""contains Poisson class"""


class Poisson():
    """Poisson distribution function"""
    def __init__(self, data=None, lambtha=1.):
        """init methoed for Poisson method"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        elif type(data) is not list:
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")
        else:
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, k):
        """finds factorial"""
        fact = 1
        for x in range(1, k + 1):
            fact = fact * x
        return fact

    def pmf(self, k):
        """probability mass function"""
        e = 2.7182818285
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)
        n = (e ** (self.lambtha * -1.0)) * (self.lambtha ** k)
        return n / self.factorial(k)

    def cdf(self, k):
        """cumulative probability function"""
        summ = 0
        if k < 1:
            return 0
        if type(k) is not int:
            k = int(k)
        for x in range(k + 1):
            summ += self.pmf(x)
        return summ
