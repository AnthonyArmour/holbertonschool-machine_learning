#!/usr/bin/env python3
"""finds the indefinite integral of polynomial"""


def poly_integral(poly, C=0):
    """finds indefinite integral of polynomial"""
    if type(poly) is not list or type(C) is not int or len(poly) < 1:
        return None
    integral = [C]
    for i, coef in enumerate(poly):
        val = coef / (i + 1)
        if val.is_integer():
            integral.append(int(val))
        else:
            integral.append(val)
    while integral[-1] == 0 and len(integral) > 1:
        integral.pop(-1)
    return integral
