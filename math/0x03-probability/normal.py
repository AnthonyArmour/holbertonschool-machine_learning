#!/usr/bin/env python3
"""Module for Normal Distribution Class"""


class Normal():
    """Normal Probability Distribution"""
    def __init__(self, data=None, mean=0, stddev=1):
        """Normal init"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            v = sum([(i - self.mean) ** 2 for i in data]) / len(data)
            self.stddev = v ** (1.0/2)

    def z_score(self, x):
        """finds the z-score given x"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """finds x given z score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Normal Probabilty Distribution Function"""
        π = 3.1415926536
        e = 2.7182818285
        exp = -1 * ((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        div = self.stddev * ((2 * π) ** (1.0/2))
        return (e ** exp) / div

    def cdf(self, x):
        """Cumulative Distribution Function"""
        xx = (x - self.mean) / (self.stddev*(2**(1.0/2)))
        return .5 * (1 + self.erf(xx))

    def erf(self, x):
        """erf function"""
        π = 3.1415926536
        co = 2 / (π**(1.0/2))
        m = x - ((x**3)/3) + ((x**5)/10) - ((x**7)/42) + ((x**9)/216)
        return co * m
