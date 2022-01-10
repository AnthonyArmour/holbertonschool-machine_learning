#!/usr/bin/env python3
"""
Module contains generator network class.
"""


from torch import nn


class Generator(nn.Module):
    """Generator subclass"""

    def __init__(self, input_size, hidden_size, output_size):
        """Class constructor."""
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        """Forward pass function for retrieving output via input."""
        output = self.main(x)
        return output
