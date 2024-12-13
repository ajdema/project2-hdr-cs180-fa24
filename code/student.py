"""
HDR stencil code - student.py
CS 1290 Computational Photography, Brown U.
"""

import cv2
import numpy as np
import scipy.optimize
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


# ========================================================================
# RADIANCE MAP RECONSTRUCTION
# ========================================================================


def solve_g(Z, B, l, w):
    """
    Given a set of pixel values observed for several pixels in several
    images with different exposure times, this function returns the
    imaging system's response function g as well as the log film irradiance
    values for the observed pixels.

    Args:
        Z[i,j]: the pixel values of pixel location number i in image j.
        B[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                (will be the same value for each i within the same j).
        l       lamdba, the constant that determines the amount of
                smoothness.
        w[z]:   the weighting function value for pixel value z (where z is between 0 - 255).

    Returns:
        g[z]:   the log exposure corresponding to pixel value z (where z is between 0 - 255).
        lE[i]:  the log film irradiance at pixel location i.

    """

    N, P = Z.shape
    n_intensity = 256  

    n_equations = N * P + (n_intensity - 2) + 1 
    A = np.zeros((n_equations, n_intensity + N), dtype=np.float64)
    b = np.zeros(n_equations, dtype=np.float64)
    
    k = 0 

    for i in range(N):
        for j in range(P):
            z_ij = Z[i, j]
            weight = w[z_ij]

            A[k, z_ij] = weight  
            A[k, n_intensity + i] = -weight 
            b[k] = weight * B[i, j]  
            k += 1

    for z in range(1, n_intensity - 1):
        A[k, z - 1] = l * w[z]  # g[z-1]
        A[k, z] = -2 * l * w[z]  # -2g[z]
        A[k, z + 1] = l * w[z]  # g[z+1]
        k += 1

    A[k, 128] = 1
    b[k] = 0
    k += 1

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    g = x[:n_intensity]
    lE = x[n_intensity:]

    return g, lE


def hdr(file_names, g_red, g_green, g_blue, w, exposure_matrix, nr_exposures):
    """
    Given the imaging system's response function g (per channel), a weighting function
    for pixel intensity values, and an exposure matrix containing the log shutter
    speed for each image, reconstruct the HDR radiance map in accordance to section
    2.2 of Debevec and Malik 1997.

    Args:
        file_names:           exposure stack image filenames
        g_red:                response function g for the red channel.
        g_green:              response function g for the green channel.
        g_blue:               response function g for the blue channel.
        w[z]:                 the weighting function value for pixel value z
                              (where z is between 0 - 255).
        exposure_matrix[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                              (will be the same value for each i within the same j).
        nr_exposures:         number of images / exposures

    Returns:
        hdr:                  the hdr radiance map.
    """
    images = []
    for fn in file_names:
        image = cv2. imread (fn)
        image = cv2. cvtColor (image, cv2. COLOR_BGR2RGB)
        images.append(image)
    images = np.array(images, dtype=np.float32)
    height, width, channels = images[0].shape
    hdr = np.zeros((height, width, channels), dtype=np.float32)
    exposure_times = exposure_matrix[0, :]

    g_channels = [g_red, g_green, g_blue]
    for c in range(channels):
        g = g_channels[c]
        channel_hdr = np.zeros((height, width), dtype=np.float32)
        weights_sum = np.zeros((height, width), dtype=np.float32)

        for j in range(len(images)):
            Z = images[j, :, :, c]  
            w_z = w[Z.astype(np.int32)]  
            g_z = g[Z.astype(np.int32)]  

            channel_hdr += w_z * (g_z - exposure_times[j])
            weights_sum += w_z

        valid_pixels = weights_sum > 0
        channel_hdr[valid_pixels] /= weights_sum[valid_pixels]
        channel_hdr[~valid_pixels] = 0  

        hdr[:, :, c] = np.exp(channel_hdr)

    return hdr

# ========================================================================
# TONE MAPPING
# ========================================================================


def tm_global_simple(hdr_radiance_map):
    """
    Simple global tone mapping function (Reinhard et al.)

    Equation:
        E_display = E_world / (1 + E_world)

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """
    tone_mapped = hdr_radiance_map / (1 + hdr_radiance_map)
    
    tone_mapped = (tone_mapped - tone_mapped.min()) / (tone_mapped.max() - tone_mapped.min())
    return tone_mapped


def tm_durand(hdr_radiance_map):
    """
    Your implementation of:
    http://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """

    hdr_copy = np.copy(hdr_radiance_map)
    hdr_copy[hdr_copy == 0] = np.min(hdr_copy[hdr_copy > 0]) * 1e-4

    intensity = np.mean(hdr_copy, axis=-1)
    intensity = np.maximum(intensity, 1e-8)
    chrominance = hdr_copy / intensity[..., None]

    log_intensity = np.log2(intensity)
    base_layer = cv2.bilateralFilter(log_intensity.astype(np.float32), d=5, sigmaColor=25, sigmaSpace=2)
    detail_layer = log_intensity - base_layer

    offset = np.percentile(base_layer, 95)
    scale = 3 / (np.percentile(base_layer, 95) - np.percentile(base_layer, 5))
    scaled_base = (base_layer - offset) * scale

    output_intensity = 2 ** (scaled_base + detail_layer)
    output_intensity = np.clip(output_intensity, 0, 10)
    output_intensity = np.log1p(output_intensity)
    output_rgb = output_intensity[..., None] * chrominance

    gamma = 1.5
    gamma_compressed = np.clip(output_rgb ** (1 / gamma), 0, 1)
    return gamma_compressed
