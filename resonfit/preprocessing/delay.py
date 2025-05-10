"""
Cable delay correction for resonator S21 data.

This module provides functionality to correct for the phase delay caused by
cables in the measurement setup.
"""

import numpy as np
from scipy.optimize import differential_evolution
from typing import Tuple, Optional, Union

from resonfit.core.base import BasePreprocessor


class CableDelayCorrector(BasePreprocessor):
    """
    Cable delay correction preprocessor.
    
    This preprocessor corrects for the phase delay caused by cables in the
    measurement setup. It optimizes the delay value so that the S21 data
    forms a circle in the complex plane.
    
    Attributes
    ----------
    optimal_delay : float or None
        Optimal cable delay value (seconds)
    """
    
    def __init__(self, bounds: Optional[Tuple[float, float]] = None):
        """
        Initialize the cable delay corrector.
        
        Parameters
        ----------
        bounds : tuple, optional
            (min_delay, max_delay) search bounds in seconds. If None, bounds
            will be calculated from the frequency span.
        """
        self.optimal_delay = None
        self.bounds = bounds
        self._optimization_result = None
        self._fit_details = None
    
    def preprocess(self, freqs: np.ndarray, s21: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct for cable delay in S21 data.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
        s21 : array_like
            Complex S21 data
            
        Returns
        -------
        tuple
            (freqs, s21_corrected) where s21_corrected has the cable delay removed
        """
        freqs = np.asarray(freqs)
        s21 = np.asarray(s21)
        
        # Calculate bounds if not provided
        if self.bounds is None:
            freq_span = np.max(freqs) - np.min(freqs) if len(freqs) > 1 else 0.0
            if freq_span == 0:
                max_delay = 1e-9  # Arbitrary small value
            else:
                max_delay = 1.0 / freq_span
            bounds_final = (-max_delay, max_delay)
        else:
            bounds_final = self.bounds
        
        # Define objective function for optimization
        def objective_function(delay_param):
            delay = delay_param[0]
            s21_corrected = s21 * np.exp(1j * 2 * np.pi * freqs * delay)
            _, _, _, error = self._fit_circle(s21_corrected)
            
            if np.isnan(error):
                return 1e10
            
            # Add a small regularization to prevent unreasonably large delays
            regularization = 1e-10 * np.abs(delay)**2
            return error + regularization
        
        # Optimize delay using differential evolution
        result = differential_evolution(
            objective_function,
            bounds=[bounds_final],
            strategy='best1bin', popsize=15, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.7, seed=None, polish=True
        )
        
        self.optimal_delay = result.x[0]
        self._optimization_result = result
        
        # Apply the optimal delay
        s21_corrected = s21 * np.exp(1j * 2 * np.pi * freqs * self.optimal_delay)
        
        # Store fit details for evaluation
        self._fit_details, _ = self._evaluate_delay(self.optimal_delay, s21)
        
        return freqs, s21_corrected
    
    def _fit_circle(self, z: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
        """
        Fit a circle to complex data using algebraic method.
        
        Parameters
        ----------
        z : array_like
            Complex data to fit
        weights : array_like, optional
            Weights for each data point
            
        Returns
        -------
        tuple
            (xc, yc, r, error) - center coordinates, radius, and fit error
        """
        x, y = z.real, z.imag
        if weights is None:
            weights = np.ones(len(x))
        
        # Build the matrices for least squares fit
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x**2 + y**2
        
        # Apply weights
        ATW = A.T * weights
        ATWA = ATW @ A
        ATWb = ATW @ b
        
        try:
            # Solve the linear system
            c_params = np.linalg.solve(ATWA, ATWb)
            
            # Extract circle parameters
            xc = c_params[0] / 2.0
            yc = c_params[1] / 2.0
            sqrt_arg = c_params[2] + xc**2 + yc**2
            if sqrt_arg < 0:
                sqrt_arg = np.abs(sqrt_arg)  # Ensure positive
            r = np.sqrt(sqrt_arg)
            
            # Calculate the error
            distances = np.sqrt((x - xc)**2 + (y - yc)**2)
            error = np.sqrt(np.sum(weights * (distances - r)**2) / np.sum(weights))
            
            return xc, yc, r, error
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan, 1e10
    
    def _evaluate_delay(self, delay: float, s21: np.ndarray) -> Tuple[dict, np.ndarray]:
        """
        Evaluate the quality of a given delay value.
        
        Parameters
        ----------
        delay : float
            Delay value to evaluate (seconds)
        s21 : array_like
            Original complex S21 data
            
        Returns
        -------
        tuple
            (fit_details, s21_corrected) - dictionary of fit details and corrected S21 data
        """
        s21_corrected = s21 * np.exp(1j * 2 * np.pi * np.asarray(self.freqs) * delay)
        
        # Calculate weights based on proximity to resonance
        # Implementation simplified from the original for clarity
        weights = np.ones(len(s21))
        
        # Fit circle to corrected data
        xc, yc, r, error = self._fit_circle(s21_corrected, weights)
        
        # Find the resonance point (minimum |S21|)
        idx_min_abs = np.argmin(np.abs(s21_corrected)) if len(s21_corrected) > 0 else 0
        
        fit_details = {
            'xc': xc, 'yc': yc, 'radius': r, 'error': error,
            'resonance_idx': idx_min_abs, 
            'weights': weights
        }
        
        return fit_details, s21_corrected
    
    @property
    def name(self):
        """Return the name of the preprocessor."""
        return "CableDelayCorrector"
    
    @property
    def parameters(self):
        """Return the parameters of the preprocessor."""
        return {
            "optimal_delay": self.optimal_delay,
            "bounds": self.bounds,
            "optimization_error": self._optimization_result.fun if self._optimization_result else None,
            "circle_center_x": self._fit_details.get('xc') if self._fit_details else None,
            "circle_center_y": self._fit_details.get('yc') if self._fit_details else None,
            "circle_radius": self._fit_details.get('radius') if self._fit_details else None,
        }
    
    @property
    def freqs(self):
        """Return the frequency array used in the last preprocessing."""
        if not hasattr(self, '_freqs'):
            return None
        return self._freqs
    
    @freqs.setter
    def freqs(self, value):
        """Set the frequency array."""
        self._freqs = value
