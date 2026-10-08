# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys

# Allow helpers package to be found when this module is imported standalone
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import load_image, get_data_path

SPHINX_IMAGE = "2560px-Great_Sphinx_of_Giza_-_20080716a.jpg"

def read_image():
    """Load and preprocess the Sphinx image: scale by 0.5 and convert to greyscale."""
    image_path = get_data_path(SPHINX_IMAGE)
    return load_image(image_path, scale_factor=2, as_gray=True)

from scipy.signal import convolve

# Skeleton Code will generate a 1D Gaussian Kernel for given sigma.
# You need to implement 2D convolution as efficiently as possible.
class GaussianFilt:
    def __init__(self, sigma):
        self.sigma = sigma

    def gauss_kernel(self):
        """Generate a normalised 1D Gaussian kernel of appropriate size for self.sigma.

        Returns:
            gauss_1d (ndarray): Shape (k_size, 1) — column vector kernel.
        """
        k_size = int(6 * self.sigma + 1)
        x = np.arange(k_size) - (k_size - 1) / 2.0
        kernel = np.exp(-0.5 * (x / self.sigma) ** 2) / (np.sqrt(2 * np.pi) * self.sigma)
        kernel = kernel / np.sum(kernel)
        return np.expand_dims(kernel, axis=1).astype(np.float64)

    def my_conv_method(self, image):
        """Perform separable 2D Gaussian convolution via two 1D passes.

        Args:
            image (ndarray): 2D greyscale image (H, W), any numeric dtype.

        Returns:
            conv_img (ndarray): Blurred image of same shape, dtype uint8.
        """
        kernel_1d = self.gauss_kernel()
        k_size = kernel_1d.shape[0]
        pad = k_size // 2

        # Pad image once before either pass to keep output size equal to input
        padded = np.pad(image, pad, mode="constant")

        # Two 1D separable convolution passes in 'valid' mode
        pass1 = convolve(padded, kernel_1d, mode="valid")
        pass2 = convolve(pass1, kernel_1d.T, mode="valid")

        # Round, clip to valid uint8 range [0, 255], and return as uint8
        return np.clip(np.round(pass2), 0, 255).astype(np.uint8)

