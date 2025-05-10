"""
Amplitude and phase normalization for resonator data.

This module provides the AmplitudePhaseNormalizer class for normalizing
S21 data such that the off-resonant point is at (1,0) in the complex plane.
"""

import numpy as np
from scipy.optimize import curve_fit

from resonfit.core.base import BasePreprocessor
from resonfit.core.models import phase_model
from resonfit.preprocessing.base import fit_circle_algebraic, calculate_weights


class AmplitudePhaseNormalizer(BasePreprocessor):
    """
    Preprocessor for amplitude and phase normalization.
    
    This class normalizes S21 data such that the off-resonant point
    is at (1,0) in the complex plane, which is the standard form
    for resonator fitting methods.
    
    Attributes
    ----------
    weight_bandwidth_scale : float
        Scale factor for the weight bandwidth used in fitting
    normalization_params : dict
        Parameters determined during normalization (circle center,
        radius, off-resonant point, etc.)
    """
    
    def __init__(self, weight_bandwidth_scale=1.0):
        """
        Initialize the amplitude and phase normalizer.
        
        Parameters
        ----------
        weight_bandwidth_scale : float, optional
            Scale factor for the weight bandwidth used in fitting, by default 1.0
        """
        self.weight_bandwidth_scale = weight_bandwidth_scale
        self.normalization_params = {}
    
    def preprocess(self, freqs, s21):
        """
        Normalize S21 data.
        
        Fits a circle to the S21 data, identifies the off-resonant point,
        and normalizes the data such that this point is at (1,0).
        
        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
        s21 : array_like
            Complex S21 data
            
        Returns
        -------
        tuple
            (freqs, s21_normalized) where s21_normalized has amplitude and phase normalized
        """
        freqs = np.asarray(freqs)
        s21 = np.asarray(s21)
        
        # Find resonance frequency for weighting
        idx_min = np.argmin(np.abs(s21))
        fr_estimate = freqs[idx_min]
        
        # Fit circle with weights focused on resonance
        weights = calculate_weights(freqs, fr_estimate, self.weight_bandwidth_scale)
        xc, yc, radius, _ = fit_circle_algebraic(s21, weights)
        
        if np.isnan(xc):
            raise ValueError("Circle fit failed in normalization. Cannot proceed.")
        
        # Center the data and unwrap phase
        s21_centered = s21 - (xc + 1j * yc)
        phase_centered = np.angle(s21_centered)
        
        # Fit phase model to determine resonance frequency and other parameters
        try:
            # Initial parameter estimates
            Ql_est = fr_estimate / ((np.max(freqs) - np.min(freqs)) / 10.0) if np.max(freqs) > np.min(freqs) else 1000.0
            theta0_est = 0.0
            
            # Bounds for phase model fitting
            lower_bounds = [np.min(freqs), 10, -np.inf]
            upper_bounds = [np.max(freqs), 1e8, np.inf]
            
            # Fit phase model
            phase_params, _ = curve_fit(
                phase_model, freqs, phase_centered,
                p0=[fr_estimate, Ql_est, theta0_est],
                bounds=(lower_bounds, upper_bounds),
                maxfev=5000
            )
            
            fr_phase, Ql_phase, theta0_phase = phase_params
            
        except (RuntimeError, ValueError) as e:
            print(f"Warning: Phase fitting failed: {e}. Using estimates.")
            fr_phase, Ql_phase, theta0_phase = fr_estimate, Ql_est, theta0_est
        
        # Calculate off-resonant point
        beta = (theta0_phase + np.pi) % (2 * np.pi)
        P_off = (xc + 1j*yc) + radius * np.exp(1j*beta)
        
        # Normalize data
        a_norm = np.abs(P_off)
        alpha_norm = np.angle(P_off)
        if a_norm == 0:
            a_norm = 1.0  # Prevent division by zero
        
        s21_normalized = s21 / (a_norm * np.exp(1j * alpha_norm))
        
        # Store parameters for reference
        self.normalization_params = {
            'xc': xc, 'yc': yc, 'radius': radius,
            'fr_phase': fr_phase, 'Ql_phase': Ql_phase, 'theta0_phase': theta0_phase,
            'a_norm': a_norm, 'alpha_norm': alpha_norm, 'P_off': P_off
        }
        
        return freqs, s21_normalized
    
    def __str__(self):
        """String representation with normalization status."""
        if self.normalization_params:
            a = self.normalization_params.get('a_norm', 'unknown')
            alpha = self.normalization_params.get('alpha_norm', 'unknown')
            return f"AmplitudePhaseNormalizer(a={a}, alpha={alpha})"
        return "AmplitudePhaseNormalizer(not normalized)"
