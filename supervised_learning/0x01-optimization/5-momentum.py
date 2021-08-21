#!/usr/bin/env python3
"""
   Module contains update_variables_momentum(alpha, beta1, var, grad, v)
   function.
"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
       Updates a variable using the gradient descent
       with momentum optimization algorithm.

       Args:
         alpha: learning rate
         beta1: momentum weight
         var: numpy.ndarray - contains the variable to be updated
         grad: numpy.ndarray - contains the gradient of var
         v: previous first moment of var

       Returns:
         The updated variable and the new moment, respectively.
    """
