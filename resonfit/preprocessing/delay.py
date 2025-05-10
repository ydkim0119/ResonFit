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
    optimization_result : OptimizeResult
        The result object from scipy.optimize.differential_evolution.
    final_circle_params : dict
        Dictionary containing the circle parameters (xc, yc, radius, error)
        for the S21 data corrected with the optimal_delay.
    final_weights : array_like
        Weights used for the circle fit with the optimal_delay.
    final_fr_estimate_for_weights : float
        Resonance frequency estimate used for calculating final_weights.
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
        self.final_circle_params = {}
        self.final_weights = None
        self.final_fr_estimate_for_weights = None

    def _objective_function_delay(self, delay_param, freqs, s21_original):
        """Internal objective function for delay optimization."""
        delay = delay_param[0]
        s21_corrected_iter = s21_original * np.exp(1j * 2 * np.pi * freqs * delay)
        
        idx_min = np.argmin(np.abs(s21_corrected_iter))
        fr_estimate = freqs[idx_min]
        
        weights = calculate_weights(freqs, fr_estimate, self.weight_bandwidth_scale)
        _, _, _, weighted_error = fit_circle_algebraic(s21_corrected_iter, weights)
        
        if np.isnan(weighted_error):
            return 1e10

        regularization = 1e-10 * np.abs(delay)**2
        return weighted_error + regularization

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
        freqs_arr = np.asarray(freqs)
        s21_arr = np.asarray(s21)
        
        if self.bounds is None:
            freq_span = np.max(freqs_arr) - np.min(freqs_arr) if len(freqs_arr) > 1 else 0.0
            if freq_span == 0:
                # Avoid division by zero if freq_span is 0, use a small default max_delay
                max_delay = 100e-9 # 100 ns, a reasonably large delay
            else:
                max_delay = 1.0 / freq_span # Heuristic for max delay
            bounds_final = (-max_delay, max_delay)
        else:
            bounds_final = self.bounds

        result = differential_evolution(
            self._objective_function_delay,
            args=(freqs_arr, s21_arr),
            bounds=[bounds_final],
            strategy='best1bin', popsize=15, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.7, seed=None, polish=True
        )

        self.optimal_delay = result.x[0]
        self.optimization_result = result
        
        s21_corrected = s21_arr * np.exp(1j * 2 * np.pi * freqs_arr * self.optimal_delay)
        
        # Store final weights and circle parameters for the optimal delay
        idx_min_corrected = np.argmin(np.abs(s21_corrected))
        self.final_fr_estimate_for_weights = freqs_arr[idx_min_corrected]
        self.final_weights = calculate_weights(freqs_arr, self.final_fr_estimate_for_weights, self.weight_bandwidth_scale)
        xc, yc, r, error = fit_circle_algebraic(s21_corrected, self.final_weights)
        self.final_circle_params = {'xc': xc, 'yc': yc, 'radius': r, 'error': error}
        
        return freqs_arr, s21_corrected
    
    def get_delay(self):
        """
        Get the optimal cable delay.
        
        Returns
        -------
        float
            Optimal cable delay (seconds)
            
        Raises
        ------
        ValueError
            If the delay has not been optimized yet
        """
        if self.optimal_delay is None:
            raise ValueError("Delay has not been optimized yet. Run preprocess() first.")
        return self.optimal_delay

    def get_final_params_for_plotting(self):
        """
        Returns parameters useful for detailed plotting of the delay correction step.

        Returns
        -------
        dict
            A dictionary containing 'weights', 'fr_estimate_for_weights', and 'circle_params'.
        """
        if self.optimal_delay is None:
            raise ValueError("Delay has not been optimized yet. Run preprocess() first.")
        return {
            'weights': self.final_weights,
            'fr_estimate_for_weights': self.final_fr_estimate_for_weights,
            'circle_params': self.final_circle_params
        }

    def __str__(self):
        """String representation with optimization status."""
        status = f"optimal_delay={self.optimal_delay*1e9:.3f} ns" if self.optimal_delay is not None else "not optimized"
        return f"CableDelayCorrector({status})"