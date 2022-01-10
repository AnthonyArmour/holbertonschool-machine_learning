#!/usr/bin/env python3
"""
Trains a Generative Adversarial Network to model a normal distribution.
"""


import torch
from scipy.stats import shapiro
from numpy.random import seed


seed(1)

def normality_test(data):
    _, p = shapiro(data)
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    return p


def train_gan(return_model=False, mbatchSize=512):
    """
    Function initializes and iterates over a GAN training algorithm.

    Args:
        return_model: Bool - return generator model if True,
        else returns 100 samples from the generator distribution.
        mbatchSize: batch size parameter for discriminator and generator training functions.

    Return:
        Generator model if return_model parameter is True, else generator
        distribution.
    """

    torch.manual_seed(111)
    steps = dInputSize = 20
    Generator = __import__('0-generator').Generator(1,16,1)
    Discriminator = __import__('1-discriminator').Discriminator(dInputSize,16,1)
    train_generator = __import__('4-train_generator').train_gen
    train_discriminator = __import__('3-train_discriminator').train_dis
    optimizer_g = torch.optim.SGD(
        Generator.parameters(), lr=1e-3, momentum=0.9)
    optimizer_d = torch.optim.SGD(
        Discriminator.parameters(), lr=1e-3, momentum=0.9)
    loss = torch.nn.BCELoss()

    for iter in range(5000):
        if iter % 200 == 0:
            print("*****Training Progress - iterations complete: {}*****".format(iter))
        if iter % 50 == 0:
            latent_space_samples = torch.randn(100, 1)
            values_test = Generator(latent_space_samples).detach()
            p = normality_test(values_test)
            if p > 0.05:
                break

        train_discriminator(Generator, Discriminator, dInputSize, 1, mbatchSize, steps, optimizer_d, loss)
        train_generator(Generator, Discriminator, 1, dInputSize, mbatchSize, steps, optimizer_g, loss)

    if return_model:
        return Generator
    else:
        latent_space_samples = torch.randn(100, 1)
        return Generator(latent_space_samples).detach()
