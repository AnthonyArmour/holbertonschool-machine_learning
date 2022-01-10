#!/usr/bin/env python3
"""
Module contains training function for GAN.
"""


import torch


def train_dis(Gen, Dis, dInputSize, gInputSize, mbatchSize, steps, optimizer, crit):
    """
    Training function for GAN.

    Args:
        Gen: Generator object.
        Dis: Discriminator object.
        dInputSize: Input size of Discriminator input data.
        gInputSize: Input size of Generator input data.
        mbatchSize: Batch size for training.
        steps: Number of steps for training.
        optimizer: Stochastic gradient descent optimizer object.
        crit: BCEloss function.

    Return:
        Error estimate of the fake and real data, along with the fake and real
        data sets of type torch.Tensor().
    """

    sample_Z = __import__('2-sample_Z').sample_Z

    for epoch in range(steps):
            # Data for training the Discriminator
            real_samples = sample_Z(0.0, 1.0, "D", dInputSize, gInputSize, mbatchSize=mbatchSize)
            real_labels = torch.ones((mbatchSize, 1))

            generated_list = []

            for i in range(mbatchSize):
                latent_space_samples = sample_Z(0.0, 1.0, "G", dInputSize, gInputSize)
                generated_samples = Gen(latent_space_samples)
                generated_list.append(generated_samples)

            generated_samples = torch.stack(generated_list, axis=0).reshape(mbatchSize, dInputSize)

            generated_labels = torch.zeros((mbatchSize, 1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_labels = torch.cat((real_labels, generated_labels))

            # Training the Discriminator
            Dis.zero_grad()
            output_Dis = Dis(all_samples)
            discriminator_loss = crit(
                output_Dis, all_labels)
            discriminator_loss.backward()
            optimizer.step()

    return discriminator_loss, all_samples