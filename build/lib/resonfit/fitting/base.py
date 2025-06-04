"""
Base functionality for fitting modules.

This module provides helper functions and base classes used by various
fitting modules.
"""

import numpy as np
from scipy.optimize import least_squares
from resonfit.preprocessing.base import calculate_weights


def estimate_initial_parameters(freqs, s21):
    """
    Estimate initial parameters for resonator fitting from S21 data.

    Parameters
    ----------
    freqs : array_like
        Frequency data (Hz)
    s21 : array_like
        Complex S21 data

    Returns
    -------
    dict
        Dictionary of initial parameter estimates (fr, Ql, etc.)
    """
    # Find resonance frequency (minimum amplitude)
    idx_min = np.argmin(np.abs(s21))
    fr_est = freqs[idx_min]

    # Estimate bandwidth from half-width at sqrt(2)/2 of the minimum-to-maximum amplitude range
    s21_mag = np.abs(s21)
    min_amp = s21_mag[idx_min]
    max_amp = np.max(s21_mag)
    half_level = min_amp + (max_amp - min_amp) / (2*np.sqrt(2))

    # Find points closest to half level on both sides of resonance
    left_idx = np.where(freqs < fr_est)[0]
    right_idx = np.where(freqs > fr_est)[0]

    if len(left_idx) > 0 and len(right_idx) > 0:
        left_half_idx = left_idx[np.argmin(np.abs(s21_mag[left_idx] - half_level))]
        right_half_idx = right_idx[np.argmin(np.abs(s21_mag[right_idx] - half_level))]
        bandwidth = freqs[right_half_idx] - freqs[left_half_idx]
        Ql_est = fr_est / bandwidth if bandwidth > 0 else 1000.0
    else:
        # Fallback: use a fraction of the frequency span
        bandwidth = (np.max(freqs) - np.min(freqs)) / 10.0
        Ql_est = fr_est / bandwidth if bandwidth > 0 else 1000.0

    # Estimate circle parameters using the minimum point and the far off points
    # Assume far off points are close to (1,0) for a normalized resonator
    dist_from_1_0 = np.abs(s21 - 1.0)
    idx_max_dist = np.argmax(dist_from_1_0)

    center_est_x = (1.0 + s21[idx_max_dist].real) / 2.0
    center_est_y = s21[idx_max_dist].imag / 2.0

    radius_est = np.sqrt((s21[idx_max_dist].real - 1.0)**2 + s21[idx_max_dist].imag**2) / 2.0

    # Estimate diameter for DCM (approximate)
    diameter_est = 2 * radius_est
    Qc_mag_est = Ql_est / diameter_est if diameter_est > 0 else 2*Ql_est

    # Estimate impedance mismatch angle
    if radius_est > 0:
        phi_est = np.arcsin(center_est_y / radius_est)
        if np.isnan(phi_est):
            phi_est = 0.0
    else:
        phi_est = 0.0

    # Return the estimates
    return {
        'fr': fr_est,
        'Ql': Ql_est,
        'Qc_mag': Qc_mag_est,
        'phi': phi_est,
        'center_x': center_est_x,
        'center_y': center_est_y,
        'radius': radius_est
    }


def evaluate_fit_quality(freqs, s21, s21_model):
    """
    Evaluate the quality of a resonator fit.

    Parameters
    ----------
    freqs : array_like
        Frequency data (Hz)
    s21 : array_like
        Complex S21 data
    s21_model : array_like
        Model S21 data from fitting

    Returns
    -------
    dict
        Dictionary containing various error metrics
    """
    # Calculate residuals
    residuals = s21 - s21_model

    # Calculate RMS error (complex)
    rms_error = np.sqrt(np.mean(np.abs(residuals)**2))

    # Calculate separate errors for magnitude and phase
    mag_residuals = np.abs(s21) - np.abs(s21_model)
    phase_residuals = np.angle(s21) - np.angle(s21_model)
    # Unwrap phase residuals to handle discontinuities
    phase_residuals = np.mod(phase_residuals + np.pi, 2*np.pi) - np.pi

    rms_mag_error = np.sqrt(np.mean(mag_residuals**2))
    rms_phase_error = np.sqrt(np.mean(phase_residuals**2))

    # Find the maximum errors
    max_mag_error = np.max(np.abs(mag_residuals))
    max_phase_error = np.max(np.abs(phase_residuals))

    # Find error at resonance
    idx_min = np.argmin(np.abs(s21))
    res_error = np.abs(residuals[idx_min])

    return {
        'rms_error': rms_error,
        'rms_mag_error': rms_mag_error,
        'rms_phase_error': rms_phase_error,
        'max_mag_error': max_mag_error,
        'max_phase_error': max_phase_error,
        'resonance_error': res_error,
        'residuals': residuals
    }
