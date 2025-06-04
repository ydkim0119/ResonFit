"""
Model functions for resonator fitting.

This module provides mathematical models for the complex transmission response
of microwave resonators, which are used in the fitting process.
"""

import numpy as np


def fano_model(f, fr, gamma, q, a):
    """
    Fano resonance model function.

    Used for initial resonance frequency estimation by fitting to |S21|^2 data.

    Parameters
    ----------
    f : array_like
        Frequency array (Hz)
    fr : float
        Resonance frequency (Hz)
    gamma : float
        Resonance width (related to FWHM)
    q : float
        Fano parameter (asymmetry coefficient)
    a : float
        Amplitude scaling factor

    Returns
    -------
    array_like
        |S21|^2 values at the given frequencies according to Fano model
    """
    return a * ((q * gamma / 2 + f - fr)**2) / ((f - fr)**2 + (gamma / 2)**2)


def phase_model(f, fr, Ql, theta0):
    """
    Phase model function for resonator transmission.

    Used during amplitude/phase correction for fitting phase data.

    Parameters
    ----------
    f : array_like
        Frequency array (Hz)
    fr : float
        Resonance frequency (Hz)
    Ql : float
        Loaded quality factor
    theta0 : float
        Background phase offset (radians)

    Returns
    -------
    array_like
        Phase values at the given frequencies (radians)
    """
    return theta0 + 2 * np.arctan(2 * Ql * (1 - f / fr))


def dcm_model(f, fr, Ql, Qc_mag, phi):
    """
    DCM (Diameter Correction Method) resonance model function.

    This model accounts for impedance mismatch by introducing a complex
    coupling quality factor. Used to fit normalized S21 data.

    Parameters
    ----------
    f : array_like
        Frequency array (Hz)
    fr : float
        Resonance frequency (Hz)
    Ql : float
        Loaded quality factor
    Qc_mag : float
        Magnitude of coupling quality factor (|Qc|)
    phi : float
        Impedance mismatch angle (radians)

    Returns
    -------
    array_like
        Complex S21 values at the given frequencies
    """
    Qc = Qc_mag * np.exp(-1j * phi)  # Complex coupling Q-factor
    return 1 - (Ql / Qc) / (1 + 2j * Ql * (f - fr) / fr)


def inverse_model(f, fr, Qi, Qc_star, phi):
    """
    Inverse S21 model function.

    This model works with inverted S21 data to directly fit Qi.

    Parameters
    ----------
    f : array_like
        Frequency array (Hz)
    fr : float
        Resonance frequency (Hz)
    Qi : float
        Internal quality factor
    Qc_star : float
        Modified coupling quality factor magnitude
    phi : float
        Impedance mismatch angle (radians)

    Returns
    -------
    array_like
        Complex inverted S21 values
    """
    return 1 + (Qi / (Qc_star * np.exp(1j * phi))) / (1 + 2j * Qi * (f - fr) / fr)


def cpzm_model(f, fr, Qi, Qc, Qa):
    """
    CPZM (Closest Pole and Zero Method) model function.

    This model represents the resonator using the closest pole and zero approach.

    Parameters
    ----------
    f : array_like
        Frequency array (Hz)
    fr : float
        Resonance frequency (Hz)
    Qi : float
        Internal quality factor
    Qc : float
        Coupling quality factor (real)
    Qa : float
        Asymmetry quality factor (related to impedance mismatch)

    Returns
    -------
    array_like
        Complex S21 values
    """
    return (1 + 2j * Qi * (f - fr) / fr) / (1 + Qi / Qc + 1j * Qi / Qa + 2j * Qi * (f - fr) / fr)


def calculate_Qi_from_DCM(Ql, Qc_complex):
    """
    Calculate internal quality factor using DCM parameters.

    Parameters
    ----------
    Ql : float
        Loaded quality factor
    Qc_complex : complex
        Complex coupling quality factor

    Returns
    -------
    float
        Internal quality factor
    """
    inv_Ql = 1.0 / Ql if Ql != 0 else np.inf
    term_real_inv_Qc = np.real(1.0 / Qc_complex) if Qc_complex != 0 else 0.0

    if inv_Ql <= term_real_inv_Qc or Ql == 0:
        return np.inf
    else:
        return 1.0 / (inv_Ql - term_real_inv_Qc)
