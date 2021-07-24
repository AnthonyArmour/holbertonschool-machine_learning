#!/usr/bin/env python3
"""finds the indefinite integral of polynomial"""


def poly_integral(poly, C=0):
    """finds indefinite integral of polynomial"""
    if type(poly) is not list or type(C) is not int:
        return None
    integral = [C]
    for i, coef in enumerate(poly):
        val = coef / (i + 1)
        if val.is_integer():
            integral.append(int(val))
        else:
            integral.append(val)
    return integral
