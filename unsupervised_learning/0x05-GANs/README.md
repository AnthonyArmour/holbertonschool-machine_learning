[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Generative Advesarial Network (GAN) Project
A generative adversarial network (GAN) is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues. The core idea of a GAN is based on the "indirect" training through the discriminator, another neural network that is able to tell how much an input is "realistic", which itself is also being updated dynamically. This basically means that the generator is not trained to minimize the distance to a specific image, but rather to fool the discriminator. This enables the model to learn in an unsupervised manner. The generative network generates candidates while the discriminative network evaluates them. The contest operates in terms of data distributions. Typically, the generative network learns to map from a latent space to a data distribution of interest, while the discriminative network distinguishes candidates produced by the generator from the true data distribution. The generative network's training objective is to increase the error rate of the discriminative network (fool the discriminative network).

Gans have been known to create very realistic artificial images of objects and also very convincing artwork.

![image](https://www.google.com/imgres?imgurl=https%3A%2F%2Fneurohive.io%2Fwp-content%2Fuploads%2F2018%2F07%2Fnon-existing-celebrities-generated-with-an-evolving-GAN-network.jpg&imgrefurl=https%3A%2F%2Fneurohive.io%2Fen%2Fstate-of-the-art%2Ffaster-learning-and-better-image-quality-with-evolving-generative-adversarial-networks%2F&tbnid=55o3X3bUvIYblM&vet=12ahUKEwiEiO_N0Ln2AhUUdM0KHcxkCbEQMygdegUIARCWAg..i&docid=jl_S7i25yqW6rM&w=1000&h=509&q=Generative%20Adversarial%20Network&hl=en&ved=2ahUKEwiEiO_N0Ln2AhUUdM0KHcxkCbEQMygdegUIARCWAg)

## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| scipy              | ^1.7.3  |
| PyTorch            | ^1.10.1 |

## Tasks
In this project, we were assigned to train a generative adversarial network to generate a normal distribution machine rather than images of people, things, or artwork.

### [Generator](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x05-GANs/5-train_GAN.py "Generator")
Trains a generative adversarial network to generate a normal distribution.

``` python
#!/usr/bin/env python3


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.express as px

ganTrainer = __import__('5-train_GAN').train_gan

fakeData = ganTrainer()

values = fakeData.data.storage().tolist()
fig = px.histogram(values, title="Histogram of Forged Distribution",nbins=50)
fig.show()
```

![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x05-GANs/images/Gan-sampled.jpg)
---