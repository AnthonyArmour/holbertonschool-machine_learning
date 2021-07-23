#!/usr/bin/env python3
"""finds derivative of polynomial """


def poly_derivative(poly):
    """finds derivative of polynomial"""
    if type(poly) is not list:
        return None
    elif len(poly) < 2:
        return [0]
    return [num * i for i, num in enumerate(poly[1:], start=1)]

