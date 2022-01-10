#!/usr/bin/env python3
"""
Module containing discriminator network class.
"""


import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator class"""

    def __init__(self, input_size, hidden_size, output_size):
        """Class constructor."""
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass function for retrieving output via input."""
        output = self.main(x)
        return output
