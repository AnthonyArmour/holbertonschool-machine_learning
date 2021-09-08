# This project is an introduction to convolutions and pooling

## Tools Created

* convolve_grayscale_valid(images, kernel):

> Performs a valid convolution on grayscale images.

* convolve_grayscale_same(images, kernel): 

> Performs a valid convolution on grayscale images.

* convolve_grayscale_padding(images, kernel, padding):

> Performs a valid convolution on grayscale images with custom padding.

* convolve_grayscale(images, kernel, padding='same', stride=(1, 1)): 

> Performs a valid convolution on grayscale images.

* convolve_channels(images, kernel, padding='same', stride=(1, 1)):

> Performs a convolution on images with channels.

* convolve(images, kernels, padding='same', stride=(1, 1)):

> Performs a convolution on images using multiple kernels.

* pool(images, kernel_shape, stride, mode='max'):

> Performs pooling on images.