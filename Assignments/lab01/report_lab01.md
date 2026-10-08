# Lab Report — Lab 01: Efficient Separable Gaussian Convolution

*Fill in each section below. Be specific — name the actual methods/parameters you used, since
your report is graded alongside your code.*

## Overview

The goal of this lab was to implement an efficient separable Gaussian blur on greyscale images to substantially beat the runtime of a naive 2D convolution.

## Implementation

I implemented `gauss_kernel` to generate a normalized 1D Gaussian column vector of length `int(6 * sigma + 1)` over integer offsets. In `my_conv_method`, I padded the image once by `k_size // 2` using `np.pad(..., mode='constant')`, performed two sequential 1D convolutions with `scipy.signal.convolve(..., mode='valid')` along the vertical and horizontal axes, and clipped the values to `uint8`.

## Results

My implementation completed in about 0.04–0.10 seconds compared to roughly 6.4–6.6 seconds for the baseline `convolve2d`, running at about 1% to 2% of the baseline time. As a result, it passed all pytest timing tiers down to the strictest 0.1x check, with the expected shape and `uint8` output type.

