"""
DCM (Diameter Correction Method) fitting implementation.

This module provides the DCMFitter class which implements the Diameter Correction Method
for fitting resonator S21 data to extract resonator parameters.
"""

import numpy as np
from scipy.optimize import least_squares
import warnings

from ...core.base import BaseFitter
from ...core.models import dcm_model, calculate_Qi_from_DCM


class DCMFitter(BaseFitter):
    """
    Fitter class implementing the Diameter Correction Method (DCM).
    
    The DCM method accounts for impedance mismatch in the transmission line by using
    a complex coupling quality factor, allowing accurate extraction of internal
    quality factor (Qi) from normalized S21 data.
    """
    
    def __init__(self, weight_bandwidth_scale=1.0):
        """
        Initialize the DCMFitter.
        
        Parameters
        ----------
        weight_bandwidth_scale : float, optional
            Scaling factor for the bandwidth used in weight calculation during fitting.
            Default is 1.0. Larger values distribute weights across a wider frequency
            range, smaller values concentrate weights closer to the resonance center.
        """
        self.weight_bandwidth_scale = float(weight_bandwidth_scale)
        self.fit_results = {}
        self.fr_estimate = None
        self.model_data = None
        self.weights = None
        
    def _calculate_weights(self, freqs, s21, fr_estimate=None, base_bandwidth_factor=0.1):
        """
        Calculate weights for fitting based on frequency distance from resonance.
        
        Parameters
        ----------
        freqs : array_like
            Frequency array (Hz)
        s21 : array_like
            Complex S21 data
        fr_estimate : float, optional
            Estimated resonance frequency (Hz). If None, will be estimated from data.
        base_bandwidth_factor : float, optional
            Base factor for bandwidth calculation as fraction of frequency span.
            Default is 0.1.
            
        Returns
        -------
        tuple
            (weights, fr_estimate) - Weights array and used resonance frequency
        """
        if fr_estimate is None:
            # Estimate resonance frequency from data
            dist_from_one = np.abs(s21 - 1.0) if len(s21) > 0 else np.array([])
            idx_res = np.argmax(dist_from_one) if len(dist_from_one) > 0 else 0
            fr_estimate = freqs[idx_res] if len(freqs) > idx_res else np.median(freqs)
            
        freq_diff = np.abs(freqs - fr_estimate)
        max_freq_diff = np.max(freq_diff) if len(freq_diff) > 0 else 0.0
        
        if max_freq_diff == 0:
            bandwidth_est = 1.0
        else:
            bandwidth_est = base_bandwidth_factor * max_freq_diff
        
        final_bandwidth = bandwidth_est * self.weight_bandwidth_scale
        
        if final_bandwidth == 0:
            weights = np.ones_like(freqs)
        else:
            weights = 1.0 / (1.0 + (freq_diff / final_bandwidth)**2)
            
        return weights, fr_estimate
    
    def fit(self, freqs, s21, use_weights=True, initial_params=None, **kwargs):
        """
        Fit the resonator data using the DCM method.
        
        Parameters
        ----------
        freqs : array_like
            Frequency array (Hz)
        s21 : array_like
            Complex S21 data (normalized)
        use_weights : bool, optional
            Whether to use frequency-dependent weighting, by default True
        initial_params : dict, optional
            Initial parameters for fitting:
            - fr: resonance frequency (Hz)
            - Ql: loaded quality factor
            - Qc_mag: coupling quality factor magnitude
            - phi: impedance mismatch angle (radians)
            If None, values will be estimated from data.
        **kwargs : dict
            Additional parameters:
            - base_bandwidth_factor: factor for bandwidth calculation (default: 0.1)
            - fr_estimate: preset resonance frequency estimate
            
        Returns
        -------
        dict
            Fitting results containing keys:
            - fr: resonance frequency (Hz)
            - Ql: loaded quality factor
            - Qc_mag: coupling quality factor magnitude
            - phi: impedance mismatch angle (radians)
            - Qi: internal quality factor
            - Qc_complex: complex coupling quality factor
            - rmse: root mean square error of the fit
            - weighted_fit: whether weighting was used
            - weights: weights array if weighting was used
        """
        # Validate inputs
        if len(freqs) != len(s21):
            raise ValueError("freqs and s21 must have the same length")
        if len(freqs) < 5:
            raise ValueError("Not enough data points for fitting (minimum 5)")
            
        # Get parameters from kwargs
        base_bandwidth_factor = kwargs.get('base_bandwidth_factor', 0.1)
        fr_est_input = kwargs.get('fr_estimate', None)
        
        # Calculate weights if needed
        self.weights = None
        if use_weights:
            self.weights, self.fr_estimate = self._calculate_weights(
                freqs, s21, fr_estimate=fr_est_input, 
                base_bandwidth_factor=base_bandwidth_factor
            )
        else:
            # Still estimate resonance frequency if not provided
            if fr_est_input is not None:
                self.fr_estimate = fr_est_input
            else:
                dist_from_one = np.abs(s21 - 1.0) if len(s21) > 0 else np.array([])
                idx_res = np.argmax(dist_from_one) if len(dist_from_one) > 0 else 0
                self.fr_estimate = freqs[idx_res] if len(freqs) > idx_res else np.median(freqs)
                
        # Determine initial parameters for the fit
        if initial_params is None:
            # Estimate bandwidth from data
            bw_est = self._estimate_bandwidth(freqs, s21)
            Ql_est = self.fr_estimate / bw_est if bw_est > 0 and self.fr_estimate != 0 else 10000.0
            
            # Fit a circle to estimate diameter
            try:
                xc, yc, r = self._fit_circle_algebraic(s21, self.weights)
                if np.isnan(xc) or r == 0:
                    # Use default values if circle fit fails
                    Qc_mag_est = Ql_est / 0.5
                    phi_est = 0.0
                else:
                    diameter = 2 * r
                    Qc_mag_est = Ql_est / diameter if diameter > 0 else Ql_est * 2.0
                    phi_est = np.arcsin(yc / r) if r > 0 and abs(yc/r) <= 1 else 0.0
            except Exception:
                # Default values if circle fit fails
                Qc_mag_est = Ql_est / 0.5
                phi_est = 0.0
                
            initial_params = {
                'fr': self.fr_estimate,
                'Ql': Ql_est,
                'Qc_mag': Qc_mag_est,
                'phi': phi_est
            }
        
        # Ensure all required parameters are present
        required_params = ['fr', 'Ql', 'Qc_mag', 'phi']
        missing_params = [p for p in required_params if p not in initial_params]
        if missing_params:
            raise ValueError(f"Missing required initial parameters: {missing_params}")
            
        # Get parameters and set bounds
        fr_init = initial_params['fr']
        Ql_init = initial_params['Ql']
        Qc_mag_init = initial_params['Qc_mag']
        phi_init = initial_params['phi']
        
        # Frequency bounds
        min_f, max_f = np.min(freqs), np.max(freqs)
        f_span = max_f - min_f
        
        fr_lower = min_f + f_span * 0.01
        fr_upper = max_f - f_span * 0.01
        
        if fr_lower >= fr_upper:
            median_f = np.median(freqs)
            fr_lower = median_f * 0.99
            fr_upper = median_f * 1.01
            
        # Parameter bounds
        bounds_lower = [fr_lower, 10.0, 1.0, -np.pi/2.0 - 0.2]
        bounds_upper = [fr_upper, 1e8, 1e8, np.pi/2.0 + 0.2]
        
        # Clip initial parameters to bounds
        fr_init = np.clip(fr_init, bounds_lower[0], bounds_upper[0])
        Ql_init = np.clip(Ql_init, bounds_lower[1], bounds_upper[1])
        Qc_mag_init = np.clip(Qc_mag_init, bounds_lower[2], bounds_upper[2])
        phi_init = np.clip(phi_init, bounds_lower[3], bounds_upper[3])
        
        params_init = [fr_init, Ql_init, Qc_mag_init, phi_init]
        
        # Define the residual function for least squares
        def residual_func(params):
            fr, Ql, Qc_mag, phi = params
            model = dcm_model(freqs, fr, Ql, Qc_mag, phi)
            res_complex = s21 - model
            
            if use_weights and self.weights is not None:
                w_sqrt = np.sqrt(self.weights)
                return np.hstack([res_complex.real * w_sqrt, res_complex.imag * w_sqrt])
            else:
                return np.hstack([res_complex.real, res_complex.imag])
                
        # Run the fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ls_result = least_squares(
                residual_func, params_init,
                bounds=(bounds_lower, bounds_upper),
                method='trf', ftol=1e-9, xtol=1e-9, max_nfev=10000
            )
            
        # Extract the results
        fr_fit, Ql_fit, Qc_mag_fit, phi_fit = ls_result.x
        
        # Calculate Qc complex and Qi
        Qc_complex_fit = Qc_mag_fit * np.exp(-1j * phi_fit)
        Qi_fit = calculate_Qi_from_DCM(Ql_fit, Qc_complex_fit)
        
        # Generate model data
        self.model_data = dcm_model(freqs, fr_fit, Ql_fit, Qc_mag_fit, phi_fit)
        
        # Calculate error
        residuals = s21 - self.model_data
        rmse = np.sqrt(np.mean(np.abs(residuals)**2))
        
        # Store results
        self.fit_results = {
            'fr': fr_fit,
            'Ql': Ql_fit,
            'Qc_mag': Qc_mag_fit,
            'phi': phi_fit,
            'Qi': Qi_fit,
            'Qc_complex': Qc_complex_fit,
            'rmse': rmse,
            'weighted_fit': use_weights,
            'weights': self.weights if use_weights else None,
            'fr_estimate': self.fr_estimate,
            'ls_result': ls_result,
            'initial_params': initial_params
        }
        
        return self.fit_results
    
    def get_model_data(self, freqs=None):
        """
        Return the model data for given frequencies.
        
        Parameters
        ----------
        freqs : array_like, optional
            Frequency array (Hz). If None, uses the frequencies from the fit.
            
        Returns
        -------
        array_like
            Complex S21 model data
        """
        if self.fit_results is None or not self.fit_results:
            raise ValueError("Model not fitted yet. Run fit() first.")
            
        if freqs is None:
            return self.model_data
            
        fr = self.fit_results['fr']
        Ql = self.fit_results['Ql']
        Qc_mag = self.fit_results['Qc_mag']
        phi = self.fit_results['phi']
        
        return dcm_model(freqs, fr, Ql, Qc_mag, phi)
    
    def _estimate_bandwidth(self, freqs, s21):
        """
        Estimate the 3dB bandwidth from S21 data.
        
        Parameters
        ----------
        freqs : array_like
            Frequency array (Hz)
        s21 : array_like
            Complex S21 data
            
        Returns
        -------
        float
            Estimated bandwidth (Hz)
        """
        if len(s21) == 0 or len(freqs) == 0:
            return (np.max(freqs) - np.min(freqs)) / 10.0 if len(freqs) > 1 else 1e6

        s21_mag_db = 20 * np.log10(np.abs(s21))
        idx_min = np.argmin(np.abs(s21))
        
        res_freq = freqs[idx_min]
        min_db_val = s21_mag_db[idx_min]
        
        half_power_db = min_db_val + 3.0
        
        try:
            left_indices = np.where((freqs < res_freq) & (s21_mag_db > min_db_val))[0]
            f_half_left = freqs[0]
            if len(left_indices) > 1:
                idx_left_hp = left_indices[np.argmin(np.abs(s21_mag_db[left_indices] - half_power_db))]
                f_half_left = freqs[idx_left_hp]

            right_indices = np.where((freqs > res_freq) & (s21_mag_db > min_db_val))[0]
            f_half_right = freqs[-1]
            if len(right_indices) > 1:
                idx_right_hp = right_indices[np.argmin(np.abs(s21_mag_db[right_indices] - half_power_db))]
                f_half_right = freqs[idx_right_hp]

            bw = f_half_right - f_half_left
            if bw > 0 and bw < (freqs[-1] - freqs[0]):
                return bw
        except:
            pass
        
        return (np.max(freqs) - np.min(freqs)) / 10.0 if len(freqs) > 1 else 1e6
    
    def _fit_circle_algebraic(self, z, weights=None):
        """
        Fit a circle to complex data using algebraic method.
        
        Parameters
        ----------
        z : array_like
            Complex data points
        weights : array_like, optional
            Weights for each data point
            
        Returns
        -------
        tuple
            (xc, yc, r) - Circle center x, y coordinates and radius
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
            
            return xc, yc, r
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan