"""
Amplitude and phase normalization for resonator S21 data.

This module provides functionality to normalize S21 data by correcting
amplitude and phase offsets. After normalization, the off-resonance point
should be at (1, 0) in the complex plane.
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings
from typing import Tuple, Optional, Dict, Any, Union

from resonfit.core.base import BasePreprocessor
from resonfit.core.models import phase_model


class AmplitudePhaseNormalizer(BasePreprocessor):
    """
    Amplitude and phase normalization preprocessor.
    
    This preprocessor normalizes S21 data by correcting for amplitude and
    phase offsets. After normalization, the off-resonance point is at (1, 0)
    in the complex plane.
    
    Attributes
    ----------
    a_norm : float or None
        Amplitude scaling factor
    alpha_norm : float or None
        Phase offset (rad)
    """
    
    def __init__(self):
        """Initialize the amplitude and phase normalizer."""
        self.a_norm = None
        self.alpha_norm = None
        self._phase_fit_params = None
        self._circle_fit_params = None
        self._off_resonance_point = None
    
    def preprocess(self, freqs: np.ndarray, s21: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize S21 data by correcting amplitude and phase offsets.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
        s21 : array_like
            Complex S21 data
            
        Returns
        -------
        tuple
            (freqs, s21_normalized) where s21_normalized has been normalized
        """
        freqs = np.asarray(freqs)
        s21 = np.asarray(s21)
        
        # Step 1: Fit a circle to the data
        xc, yc, radius, _ = self._fit_circle(s21)
        
        if np.isnan(xc):
            raise ValueError("Circle fit failed in normalization. Cannot proceed.")
        
        # Step 2: Center the data and extract phase
        s21_centered = s21 - (xc + 1j * yc)
        phase_centered = np.angle(s21_centered)
        
        # Step 3: Fit the phase to find resonance and offset
        try:
            fr, Ql, theta0 = self._fit_phase(freqs, phase_centered, np.abs(s21))
            self._phase_fit_params = {'fr': fr, 'Ql': Ql, 'theta0': theta0}
        except Exception as e:
            raise ValueError(f"Phase fitting failed: {str(e)}")
        
        # Step 4: Determine off-resonance point
        beta = (theta0 + np.pi) % (2 * np.pi)
        P_off = (xc + 1j*yc) + radius * np.exp(1j*beta)
        self._off_resonance_point = P_off
        
        # Step 5: Calculate normalization factors
        self.a_norm = np.abs(P_off)
        self.alpha_norm = np.angle(P_off)
        
        if self.a_norm == 0:
            warnings.warn("Amplitude normalization factor 'a' is zero. Using a=1.0 instead.")
            self.a_norm = 1.0
        
        # Step 6: Apply normalization
        s21_normalized = s21 / (self.a_norm * np.exp(1j * self.alpha_norm))
        
        # Store circle fit parameters
        self._circle_fit_params = {'xc': xc, 'yc': yc, 'radius': radius}
        
        return freqs, s21_normalized
    
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
    
    def _fit_phase(self, freqs: np.ndarray, phase_data: np.ndarray, 
                   s21_mag: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        Fit phase data to the phase model.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
        phase_data : array_like
            Phase data (rad)
        s21_mag : array_like, optional
            S21 magnitude data for resonance estimation
            
        Returns
        -------
        tuple
            (fr, Ql, theta0) - Resonance frequency, loaded quality factor, and phase offset
        """
        # Unwrap the phase to avoid discontinuities
        unwrapped_phase = np.unwrap(phase_data)
        
        # Set up bounds for fitting
        min_f, max_f = np.min(freqs), np.max(freqs)
        f_span = max_f - min_f
        
        # Set bound for resonance frequency
        fr_lower_bound = min_f + f_span * 0.01
        fr_upper_bound = max_f - f_span * 0.01
        
        if fr_lower_bound >= fr_upper_bound:
            median_f = np.median(freqs)
            fr_lower_bound = median_f - abs(median_f * 0.01)
            fr_upper_bound = median_f + abs(median_f * 0.01)
        
        # Estimate initial parameters
        if s21_mag is not None:
            idx_min = np.argmin(s21_mag)
        else:
            idx_min = np.argmax(np.abs(np.diff(unwrapped_phase))) + 1
            if idx_min >= len(freqs):
                idx_min = len(freqs) // 2
        
        fr_est = freqs[idx_min]
        fr_est = max(fr_lower_bound, min(fr_est, fr_upper_bound))
        
        # Estimate Ql based on phase slope
        Ql_est = fr_est / (f_span / 10.0)
        if len(freqs) > 10 and fr_est != 0:
            dp_df = np.gradient(unwrapped_phase, freqs)
            slope_at_res = dp_df[idx_min] if idx_min < len(dp_df) else dp_df[-1]
            Ql_est = abs(slope_at_res * fr_est / (-4.0))
        
        # Estimate theta0 from edge phases
        n_edge = max(1, min(3, len(freqs) // 20))
        edge_phases = np.concatenate([unwrapped_phase[:n_edge], unwrapped_phase[-n_edge:]])
        theta0_est = np.mean(edge_phases)
        
        initial_params = (fr_est, Ql_est, theta0_est)
        bounds = ([fr_lower_bound, 10, -np.inf], [fr_upper_bound, 1e8, np.inf])
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params, _ = curve_fit(
                    phase_model, freqs, unwrapped_phase,
                    p0=initial_params, bounds=bounds, maxfev=5000
                )
            return params
        except Exception as e:
            warnings.warn(f"Phase fitting failed with error: {str(e)}. Using initial estimates.")
            return initial_params
    
    @property
    def name(self):
        """Return the name of the preprocessor."""
        return "AmplitudePhaseNormalizer"
    
    @property
    def parameters(self):
        """Return the parameters of the preprocessor."""
        params = {
            "a_norm": self.a_norm,
            "alpha_norm": self.alpha_norm,
        }
        
        if self._phase_fit_params:
            params.update({
                "fr_phase": self._phase_fit_params.get('fr'),
                "Ql_phase": self._phase_fit_params.get('Ql'),
                "theta0_phase": self._phase_fit_params.get('theta0'),
            })
        
        if self._circle_fit_params:
            params.update({
                "circle_center_x": self._circle_fit_params.get('xc'),
                "circle_center_y": self._circle_fit_params.get('yc'),
                "circle_radius": self._circle_fit_params.get('radius'),
            })
        
        if self._off_resonance_point is not None:
            params.update({
                "P_off_real": self._off_resonance_point.real,
                "P_off_imag": self._off_resonance_point.imag,
            })
        
        return params
