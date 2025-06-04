"""
Base functionality for preprocessing modules.

This module provides helper functions and base classes used by the
various preprocessing modules.
"""

import numpy as np
from scipy.optimize import least_squares


def fit_circle_algebraic(z, weights=None):
    """
    Fit a circle to complex data (x + iy) using an algebraic method.

    Parameters
    ----------
    z : array_like
        Complex data points
    weights : array_like, optional
        Weights for each data point, by default None

    Returns
    -------
    tuple
        (xc, yc, r, error)
        - xc, yc: circle center coordinates
        - r: circle radius
        - error: weighted RMS error of the fit
    """
    x, y = z.real, z.imag
    if weights is None:
        weights = np.ones(len(x))

    A = np.column_stack([x, y, np.ones(len(x))])
    b = x**2 + y**2

    ATW = A.T * weights
    ATWA = ATW @ A
    ATWb = ATW @ b

    try:
        c_params = np.linalg.solve(ATWA, ATWb)
        xc = c_params[0] / 2.0
        yc = c_params[1] / 2.0
        sqrt_arg = c_params[2] + xc**2 + yc**2
        if sqrt_arg < 0:
            sqrt_arg = np.abs(sqrt_arg)
        r = np.sqrt(sqrt_arg)

        distances_from_center = np.sqrt((x - xc)**2 + (y - yc)**2)
        weighted_error = np.sqrt(np.sum(weights * (distances_from_center - r)**2) / np.sum(weights))
        return xc, yc, r, weighted_error
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, 1e10


def calculate_weights(freqs, fr_estimate, weight_bandwidth_scale=1.0, base_bandwidth_factor=0.1):
    """
    Calculate weights for fitting based on proximity to resonance frequency.

    Parameters
    ----------
    freqs : array_like
        Frequency data (Hz)
    fr_estimate : float
        Estimated resonance frequency (Hz)
    weight_bandwidth_scale : float, optional
        Scale factor for the weight bandwidth, by default 1.0
    base_bandwidth_factor : float, optional
        Base bandwidth as a fraction of frequency span, by default 0.1

    Returns
    -------
    array_like
        Weights for each frequency point
    """
    freq_diff = np.abs(freqs - fr_estimate)
    max_freq_diff = np.max(freq_diff) if len(freq_diff) > 0 else 0.0

    if max_freq_diff == 0:
        _bandwidth_est_base = 1.0
    else:
        _bandwidth_est_base = base_bandwidth_factor * max_freq_diff

    final_bandwidth_est = _bandwidth_est_base * weight_bandwidth_scale

    if final_bandwidth_est == 0:
        weights = np.ones_like(freqs)
    else:
        weights = 1.0 / (1.0 + (freq_diff / final_bandwidth_est)**2)
    return weights


def find_resonance_frequency(freqs, s21, try_fano=True):
    """
    Estimate the resonance frequency from S21 data.

    Parameters
    ----------
    freqs : array_like
        Frequency data (Hz)
    s21 : array_like
        Complex S21 data
    try_fano : bool, optional
        Whether to attempt a Fano model fit, by default True

    Returns
    -------
    float
        Estimated resonance frequency (Hz)
    """
    idx_min = np.argmin(np.abs(s21))
    fr_min = freqs[idx_min]

    return fr_min  # Simplify for now, Fano fitting will be added later
