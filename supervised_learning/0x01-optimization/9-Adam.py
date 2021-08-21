#!/usr/bin/env python3
"""
   Module contains
   update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
   function
"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
       Updates a variable in place using the Adam optimization algorithm.

       Args:
         alpha: learning rate
         beta1: weight used for the first moment
         beta2: weight used for the second moment
         epsilon: small number to avoid division by zero
         var: numpy.ndarray containing the variable to be updated
         grad: numpy.ndarray containing the gradient of var
         v: previous first moment of var
         s: previous second moment of var
         t: time step used for bias correction

       Returns:
         The updated variable, the new first moment, and the new second moment,
         respectively.
    """
