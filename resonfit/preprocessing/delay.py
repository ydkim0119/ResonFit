"""
Cable delay correction for resonator data.

This module provides the CableDelayCorrector class for removing
the frequency-dependent phase shift caused by cable delay.
"""

import numpy as np
from scipy.optimize import differential_evolution

from resonfit.core.base import BasePreprocessor
from resonfit.preprocessing.base import fit_circle_algebraic, calculate_weights


class CableDelayCorrector(BasePreprocessor):
    """
    Preprocessor for cable delay correction.
    
    Cable delay causes a frequency-dependent phase shift in the S21 data,
    which distorts the resonance circle in the complex plane. This class
    optimizes the delay value to make the S21 data form a circle.
    
    Attributes
    ----------
    bounds : tuple, optional
        Bounds for delay optimization (min_delay, max_delay) in seconds
    weight_bandwidth_scale : float
        Scale factor for the weight bandwidth used in fitting
    optimal_delay : float
        Optimal cable delay found after preprocessing (seconds)
    """
    
    def __init__(self, bounds=None, weight_bandwidth_scale=1.0):
        """
        Initialize the cable delay corrector.
        
        Parameters
        ----------
        bounds : tuple, optional
            Bounds for delay optimization (min_delay, max_delay) in seconds.
            If None, will be calculated from frequency span.
        weight_bandwidth_scale : float, optional
            Scale factor for the weight bandwidth used in fitting, by default 1.0
        """
        self.bounds = bounds
        self.weight_bandwidth_scale = weight_bandwidth_scale
        self.optimal_delay = None
        self.optimization_result = None
    
    def preprocess(self, freqs, s21):
        """
        Apply cable delay correction to S21 data.
        
        Optimizes the delay value to make the S21 data form a circle,
        then applies the correction.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
        s21 : array_like
            Complex S21 data
            
        Returns
        -------
        tuple
            (freqs, s21_corrected) where s21_corrected has cable delay removed
        """
        freqs = np.asarray(freqs)
        s21 = np.asarray(s21)
        
        if self.bounds is None:
            freq_span = np.max(freqs) - np.min(freqs) if len(freqs) > 1 else 0.0
            if freq_span == 0:
                max_delay = 1e-9
            else:
                max_delay = 1.0 / freq_span
            bounds_final = (-max_delay, max_delay)
        else:
            bounds_final = self.bounds

        def objective_function_delay(delay_param):
            delay = delay_param[0]
            s21_corrected_iter = s21 * np.exp(1j * 2 * np.pi * freqs * delay)
            
            # Use the magnitude minimum as a rough estimate of resonance frequency
            idx_min = np.argmin(np.abs(s21_corrected_iter))
            fr_estimate = freqs[idx_min]
            
            weights = calculate_weights(freqs, fr_estimate, self.weight_bandwidth_scale)
            _, _, _, weighted_error = fit_circle_algebraic(s21_corrected_iter, weights)
            
            if np.isnan(weighted_error):
                return 1e10

            regularization = 1e-10 * np.abs(delay)**2  # Small regularization to prefer smaller delays
            return weighted_error + regularization

        result = differential_evolution(
            objective_function_delay,
            bounds=[bounds_final],
            strategy='best1bin', popsize=15, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.7, seed=None, polish=True
        )

        self.optimal_delay = result.x[0]
        self.optimization_result = result
        
        s21_corrected = s21 * np.exp(1j * 2 * np.pi * freqs * self.optimal_delay)
        
        return freqs, s21_corrected
    
    def __str__(self):
        """String representation with optimization status."""
        status = f"optimal_delay={self.optimal_delay*1e9:.3f} ns" if self.optimal_delay is not None else "not optimized"
        return f"CableDelayCorrector({status})"
