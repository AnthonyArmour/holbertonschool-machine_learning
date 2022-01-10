#!/usr/bin/env python3
"""
Module contains sampling function that creates
input for the generator and discriminator networks.
"""


import torch


def sample_Z(mu, sigma, sampleType, dInputSize, gInputSize, mbatchSize=None):
    """
    Creates input for the generator and discriminator.

    Args:
        mu: Mean of the distribution.
        sigma: Standard deviation of the distribution.
        sampleType: String that selects which model to sample for.

    Return:
        torch.Tensor type for both generator and discriminator if the
        parameters are correct, 0 otherwise.
    """

    if sampleType == "G":
        return torch.randn((dInputSize, gInputSize))
    elif sampleType == "D":
        return torch.normal(mu, sigma, (mbatchSize, dInputSize))
    else:
        return 0