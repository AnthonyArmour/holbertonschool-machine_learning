#!/usr/bin/env python3
"""
   Module contains
   update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s)
   function
"""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
       Updates a variable using the RMSProp optimization algorithm.

       Args:
         alpha: learning rate
         beta2: RMSProp weight
         epsilon: small number to avoid division by zero
         var: numpy.ndarray - contains the variable to be updated
         grad: numpy.ndarray - contains the gradient of var
         s: previous second moment of var

       Returns:
         The updated variable and the new moment, respectively.
    """
