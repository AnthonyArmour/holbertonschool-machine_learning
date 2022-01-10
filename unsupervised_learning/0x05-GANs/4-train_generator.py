#!/usr/bin/env python3
"""
Module contains function for training generator network.
"""


import torch


def train_gen(Gen, Dis, gInputSize, dInputSize, mbatchSize, steps, optimizer, crit):
    """
    Trains generator network.

    Args:
        Gen: Generator object.
        Dis: Discriminator object.
        gInputSize: Input size of generator input data.
        mbatchSize: Batch size for training.
        steps: Number of steps for training.
        optimizer: Stochastic gradient descent optimizer object.
        crit: BCEloss function.

    Return:
        Error of the fake data, and the fake data set of type torch.Tensor()
    """

    sample_Z = __import__('2-sample_Z').sample_Z

    for epoch in range(steps):
        real_samples_labels = torch.ones((mbatchSize, gInputSize))

        generated_list = []

        for i in range(mbatchSize):
            latent_space_samples = sample_Z(0.0, 1.0, "G", dInputSize, gInputSize)
            generated_samples = Gen(latent_space_samples)
            generated_list.append(generated_samples)

        generated_samples = torch.stack(generated_list, axis=0).reshape(mbatchSize, dInputSize)

        # Training the generator
        Gen.zero_grad()
        output_discriminator_generated = Dis(generated_samples)
        loss_generator = crit(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer.step()

    return loss_generator, generated_samples