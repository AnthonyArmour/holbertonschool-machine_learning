#!/usr/bin/env python3
"""finds derivative of polynomial """


def poly_derivative(poly):
    """finds derivative of polynomial"""
    if len(poly) < 2:
        return [0]
    poly_prime = [num * i for i, num in enumerate(poly[1:], start=1)]
    return poly_prime
