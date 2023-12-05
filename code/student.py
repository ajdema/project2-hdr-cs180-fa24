"""
HDR stencil code - student.py
CS 1290 Computational Photography, Brown U.
"""

import numpy as np
import cv2
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

    g = np.random.random(256)
    lE = np.random.random((Z.shape[0] * Z.shape[1]))

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

    image = cv2.cvtColor(cv2.imread(file_names[0]), cv2.COLOR_BGR2RGB)
    hdr = np.random.random(image.shape)

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

    return np.random.random(hdr_radiance_map.shape)


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

    return np.random.random(hdr_radiance_map.shape)
