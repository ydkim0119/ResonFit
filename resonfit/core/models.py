"""
Model functions for resonator fits.

This module provides various analytical models for microwave resonators
that can be used by fitting methods.
"""

import numpy as np


def fano_model(f, fr, gamma, q, a):
    """
    Fano resonance model function.
    Used primarily to estimate initial resonance frequency from |S21|^2 data.

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
        |S21|^2 values calculated by the Fano model at given frequencies
    """
    return a * ((q * gamma / 2 + f - fr)**2) / ((f - fr)**2 + (gamma / 2)**2)


def phase_model(f, fr, Ql, theta0):
    """
    Phase model function for resonator transmission.
    Used in amplitude/phase correction stage.

    Parameters
    ----------
    f : array_like
        Frequency array (Hz)
    fr : float
        Resonance frequency (Hz)
    Ql : float
        Loaded quality factor
    theta0 : float
        Background phase offset (rad)

    Returns
    -------
    array_like
        Phase values (rad) calculated by the model at given frequencies
    """
    return theta0 + 2 * np.arctan(2 * Ql * (1 - f / fr))


def dcm_model(f, fr, Ql, Qc_mag, phi):
    """
    DCM (Diameter Correction Method) resonance model function.
    This model is appropriate for normalized S21 data where the
    off-resonance point is at (1,0) in the complex plane.

    Parameters
    ----------
    f : array_like
        Frequency array (Hz)
    fr : float
        Resonance frequency (Hz)
    Ql : float
        Loaded quality factor
    Qc_mag : float
        Magnitude of the coupling quality factor (|Qc|)
    phi : float
        Coupling impedance phase (rad, related to coupling asymmetry)

    Returns
    -------
    array_like
        Complex S21 values calculated by the DCM model at given frequencies
    """
    Qc = Qc_mag * np.exp(-1j * phi)  # Complex coupling quality factor
    return 1 - (Ql / Qc) / (1 + 2j * Ql * (f - fr) / fr)


def inverse_model(f, fr, Qi, Qc_mag, phi):
    """
    Inverse S21 model function.
    This model directly fits the inverted S21 data.

    Parameters
    ----------
    f : array_like
        Frequency array (Hz)
    fr : float
        Resonance frequency (Hz)
    Qi : float
        Internal quality factor
    Qc_mag : float
        Magnitude of the coupling quality factor (|Qc|)
    phi : float
        Coupling impedance phase (rad, related to coupling asymmetry)

    Returns
    -------
    array_like
        Complex inverted S21 values calculated by the model at given frequencies
    """
    Qc = Qc_mag * np.exp(-1j * phi)  # Complex coupling quality factor
    return 1 + (Qi / Qc) / (1 + 2j * Qi * (f - fr) / fr)


def cpzm_model(f, fr, Qi, Qc, Qa):
    """
    CPZM (Closest Pole and Zero Method) resonance model function.

    Parameters
    ----------
    f : array_like
        Frequency array (Hz)
    fr : float
        Resonance frequency (Hz)
    Qi : float
        Internal quality factor
    Qc : float
        Coupling quality factor (real number)
    Qa : float
        Asymmetry quality factor (real number, related to impedance mismatch)

    Returns
    -------
    array_like
        Complex S21 values calculated by the CPZM model at given frequencies
    """
    return (1 + 2j * Qi * (f - fr) / fr) / (1 + Qi / Qc + 1j * Qi / Qa + 2j * Qi * (f - fr) / fr)
