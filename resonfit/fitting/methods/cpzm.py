"""
CPZM (Closest Pole and Zero Method) fitting implementation.

This module provides the CPZMFitter class which implements the Closest Pole and Zero Method
for fitting resonator S21 data. This method parameterizes the resonator response
in terms of Qi, Qc (real coupling Q), and Qa (asymmetry Q).

Reference:
C. Deng et al., J. Appl. Phys. 114, 054504 (2013)
Keegan Mullins, "Highly Accurate Data Fitting for Microwave Resonators", Thesis (2021), Section 3.5
"""

import numpy as np
from scipy.optimize import least_squares
import warnings

from ...core.base import BaseFitter
from ...core.models import cpzm_model # S21 = (1 + 2j Qi df/fr) / (1 + Qi/Qc + 1j Qi/Qa + 2j Qi df/fr)

class CPZMFitter(BaseFitter):
    """
    Fitter class implementing the Closest Pole and Zero Method (CPZM).

    The CPZM method models the S21 transmission using internal (Qi),
    coupling (Qc - real valued), and asymmetry (Qa) quality factors.
    Impedance mismatch is captured by Qa.
    """

    def __init__(self, weight_bandwidth_scale=1.0):
        """
        Initialize the CPZMFitter.

        Parameters
        ----------
        weight_bandwidth_scale : float, optional
            Scaling factor for the bandwidth used in weight calculation during fitting.
            Default is 1.0.
        """
        self.weight_bandwidth_scale = float(weight_bandwidth_scale)
        self.fit_results = {}
        self.fr_estimate = None
        self.model_data = None
        self.weights = None

    def _calculate_weights(self, freqs, s21, fr_estimate=None, base_bandwidth_factor=0.1):
        """
        Calculate weights for fitting based on frequency distance from resonance.
        Weights emphasize points around the resonance dip of |S21|.
        """
        s21_mag = np.abs(s21)

        if fr_estimate is None:
            idx_res = np.argmin(s21_mag) if len(s21_mag) > 0 else 0
            fr_estimate = freqs[idx_res] if len(freqs) > idx_res else np.median(freqs)
        else: # If fr_estimate is provided, still find the index for min_mag calculation
            idx_res = np.argmin(np.abs(freqs - fr_estimate))


        freq_diff = np.abs(freqs - fr_estimate)
        
        # Estimate bandwidth based on FWHM of the |S21| dip
        min_mag = s21_mag[idx_res]
        
        # Estimate off-resonant magnitude (e.g., average of first/last 10% of points, or overall max)
        num_edge_points = max(1, len(freqs) // 10)
        if len(freqs) > 2 * num_edge_points:
            off_res_mag_est = np.mean(np.concatenate((s21_mag[:num_edge_points], s21_mag[-num_edge_points:])))
        elif len(s21_mag) > 0:
            off_res_mag_est = np.max(s21_mag) # Fallback to overall max
        else:
            off_res_mag_est = 1.0 # Default if no data

        # If min_mag is very close to off_res_mag_est (shallow dip), use a wider bandwidth estimation
        if off_res_mag_est - min_mag < 0.01 * off_res_mag_est : # if dip is less than 1% of off-res magnitude
            target_mag_level = min_mag + 0.05 * off_res_mag_est # a small step above minimum
        else:
            # Target magnitude for FWHM-like calculation: min_mag + (off_res_mag_est - min_mag)/sqrt(2) for power, or /2 for amplitude
            target_mag_level = min_mag + (off_res_mag_est - min_mag) / 2.0 # Halfway in amplitude

        bandwidth_est = (np.max(freqs) - np.min(freqs)) * base_bandwidth_factor # Default fallback
        
        try:
            # Find indices where magnitude is around target_mag_level
            left_indices = np.where((freqs < fr_estimate) & (s21_mag >= target_mag_level))[0]
            right_indices = np.where((freqs > fr_estimate) & (s21_mag >= target_mag_level))[0]

            if len(left_indices) > 0 and len(right_indices) > 0:
                # More robust: find the outermost points that are still above the target level
                # Or, find points closest to the target_mag_level
                f_low_idx = left_indices[np.argmin(np.abs(s21_mag[left_indices] - target_mag_level))]
                f_high_idx = right_indices[np.argmin(np.abs(s21_mag[right_indices] - target_mag_level))]
                
                f_low = freqs[f_low_idx]
                f_high = freqs[f_high_idx]
                
                current_bw = f_high - f_low
                if current_bw > 0:
                    bandwidth_est = current_bw
            elif len(freqs) <= 5: # Very few points, use span
                 pass # Keep default fallback
            else: # Only one side or no clear points, use fallback based on span
                pass # Keep default fallback

        except Exception: # Catch any error during bandwidth estimation
            pass # Keep default fallback

        if bandwidth_est <= 0: # Ensure bandwidth is positive
            bandwidth_est = (np.max(freqs) - np.min(freqs)) * base_bandwidth_factor if len(freqs) > 1 else 1.0
            if bandwidth_est <= 0: bandwidth_est = fr_estimate / 1000.0 # Absolute fallback if span is also zero


        final_bandwidth = bandwidth_est * self.weight_bandwidth_scale

        if final_bandwidth <= 1e-9: # Avoid division by zero or extremely small bandwidth
            weights = np.ones_like(freqs)
            # print(f"CPZM Debug: final_bandwidth too small ({final_bandwidth}), using uniform weights. fr_est={fr_estimate}, bw_est={bandwidth_est}")
        else:
            weights = 1.0 / (1.0 + (freq_diff / final_bandwidth)**2)
            # print(f"CPZM Debug: fr_est={fr_estimate}, bw_est={bandwidth_est}, final_bw={final_bandwidth}, min_w={np.min(weights)}, max_w={np.max(weights)}")
        
        return weights, fr_estimate


    def _estimate_initial_params_cpzm(self, freqs, s21):
        """Estimate initial parameters for the CPZM fit."""
        s21_mag = np.abs(s21)
        idx_min = np.argmin(s21_mag) if len(s21_mag) > 0 else 0
        fr_est = freqs[idx_min] if len(freqs) > idx_min else np.median(freqs)

        # Estimate Ql (loaded Q) from FWHM of |S21|
        min_val = s21_mag[idx_min] if len(s21_mag) > idx_min else 0.0
        
        num_edge_points = max(1, len(freqs) // 10)
        if len(freqs) > 2*num_edge_points :
            off_res_mag_est = np.mean(np.concatenate((s21_mag[:num_edge_points], s21_mag[-num_edge_points:])))
        elif len(s21_mag) > 0:
            off_res_mag_est = np.max(s21_mag)
        else:
            off_res_mag_est = 1.0

        # Ensure off_res_mag_est is greater than min_val for sensible bandwidth calculation
        if off_res_mag_est <= min_val : off_res_mag_est = min_val + 0.1 * (np.max(s21_mag) if len(s21_mag)>0 else 1.0) # Add a bit if flat
        if off_res_mag_est <= min_val : off_res_mag_est = 1.0 # Absolute fallback

        half_power_level = min_val + (off_res_mag_est - min_val) / 2.0 # Mid-amplitude
        
        bandwidth_est = (np.max(freqs)-np.min(freqs))/10.0 # Default
        try:
            left_half_indices = np.where((freqs < fr_est) & (s21_mag >= half_power_level))[0]
            right_half_indices = np.where((freqs > fr_est) & (s21_mag >= half_power_level))[0]
            
            if len(left_half_indices) > 0 and len(right_half_indices) > 0:
                f1 = freqs[left_half_indices[-1]] 
                f2 = freqs[right_half_indices[0]]
                current_bw = f2 - f1
                if current_bw > 0 : bandwidth_est = current_bw
        except:
            pass # Keep default bandwidth

        if bandwidth_est <= 0: bandwidth_est = (np.max(freqs)-np.min(freqs))/10.0
        Ql_est = fr_est / bandwidth_est if bandwidth_est > 0 and fr_est !=0 else 10000.0

        Qi_est = Ql_est # Initial guess: Qi ~ Ql
        # Rough Qc estimation based on dip depth relative to off-resonance magnitude
        # S21_res ~ (1) / (1 + Qi/Qc) if Qa is large (symmetric)
        # |S21_res| ~ 1 / (1 + Qi/Qc)  => Qi/Qc ~ 1/|S21_res| - 1
        # Qc ~ Qi / (1/|S21_res| - 1) = Qi * |S21_res| / (1 - |S21_res|)
        # Need to ensure min_val is not too close to off_res_mag_est for this to be stable
        # And also that min_val is not zero.
        if min_val > 1e-9 and off_res_mag_est > min_val : # Ensure min_val is not zero and there's a dip
            # Normalize min_val by off_res_mag_est for the formula assuming off-res is 1
            s21_res_norm_mag = min_val / off_res_mag_est
            if s21_res_norm_mag < 0.99: # Avoid division by zero if s21_res_norm_mag is close to 1
                 Qc_est = Qi_est * s21_res_norm_mag / (1.0 - s21_res_norm_mag)
            else: # If very shallow dip, assume Qc ~ Qi
                 Qc_est = Qi_est
        else: # Fallback if min_val is zero or no clear dip
            Qc_est = Qi_est * 2.0 # Default: assume undercoupled or critically coupled

        # For Qa, initial guess: large value for nearly symmetric resonance.
        # If the phase at resonance (after normalization to remove background phase) is non-zero, Qa is finite.
        # Phase_at_res_S21 ~ angle of (1 - Qi/Qc - j Qi/Qa) / (1 + Qi/Qc + j Qi/Qa)
        # If S21 is normalized so off-res is at (1,0), then S21_res value's phase depends on Qa
        # angle(s21[idx_min]) can give some hint, but it's tricky without full normalization.
        Qa_est = 10 * Qi_est # Start with a relatively symmetric line shape. A large Qa means small asymmetry.

        return {
            'fr': fr_est,
            'Qi': max(10.0, Qi_est),
            'Qc': max(10.0, Qc_est),
            'Qa': max(10.0, Qa_est) 
        }

    def fit(self, freqs, s21, use_weights=True, initial_params=None, **kwargs):
        """
        Fit the resonator data using the CPZM method.

        Parameters
        ----------
        freqs : array_like
            Frequency array (Hz)
        s21 : array_like
            Complex S21 data (ideally normalized, but CPZM can handle some baseline effects via Qa)
        use_weights : bool, optional
            Whether to use frequency-dependent weighting, by default True
        initial_params : dict, optional
            Initial parameters for fitting:
            - fr: resonance frequency (Hz)
            - Qi: internal quality factor
            - Qc: coupling quality factor (real)
            - Qa: asymmetry quality factor (real)
            If None, values will be estimated.
        **kwargs : dict
            Additional parameters for weighting or initial estimation.

        Returns
        -------
        dict
            Fitting results including fr, Qi, Qc, Qa, Ql (calculated), rmse.
        """
        if len(freqs) != len(s21):
            raise ValueError("freqs and s21 must have the same length")
        if len(freqs) < 5:
            raise ValueError("Not enough data points for fitting (minimum 5)")

        base_bandwidth_factor = kwargs.get('base_bandwidth_factor', 0.1) # Used in _calculate_weights if needed
        fr_est_input = kwargs.get('fr_estimate', None) # fr_estimate can be passed from pipeline

        self.weights = None
        if use_weights:
            self.weights, self.fr_estimate = self._calculate_weights(
                freqs, s21, fr_estimate=fr_est_input,
                base_bandwidth_factor=base_bandwidth_factor
            )
        else: # No weights, but still need fr_estimate if not provided
            if fr_est_input is not None:
                self.fr_estimate = fr_est_input
            else: # Estimate fr if not given and no weights
                idx_res = np.argmin(np.abs(s21)) if len(s21) > 0 else 0
                self.fr_estimate = freqs[idx_res] if len(freqs) > idx_res else np.median(freqs)


        if initial_params is None:
            initial_params_est = self._estimate_initial_params_cpzm(freqs, s21)
            # Override fr from estimation with fr_estimate from weighting if available and more reliable
            if self.fr_estimate : initial_params_est['fr'] = self.fr_estimate
            initial_params = initial_params_est


        required_params = ['fr', 'Qi', 'Qc', 'Qa']
        missing_params = [p for p in required_params if p not in initial_params]
        if missing_params:
            # Try to re-estimate if some are missing but fr_estimate is good
            if self.fr_estimate and 'fr' not in initial_params: initial_params['fr'] = self.fr_estimate
            if self.fr_estimate and ('Qi' not in initial_params or 'Qc' not in initial_params or 'Qa' not in initial_params):
                temp_est = self._estimate_initial_params_cpzm(freqs, s21)
                for p in ['Qi', 'Qc', 'Qa']:
                    if p not in initial_params: initial_params[p] = temp_est[p]
            # Final check
            missing_params = [p for p in required_params if p not in initial_params]
            if missing_params:
                 raise ValueError(f"Missing required initial parameters for CPZMFitter: {missing_params}. Current initial_params: {initial_params}")


        fr_init, Qi_init, Qc_init, Qa_init = initial_params['fr'], initial_params['Qi'], \
                                             initial_params['Qc'], initial_params['Qa']

        min_f, max_f = np.min(freqs), np.max(freqs)
        f_span = max_f - min_f if len(freqs) > 1 else 0
        
        # Fr bounds should be reasonably within the data span
        fr_lower_bound = min_f + f_span * 0.01 if f_span > 0 else min_f * 0.99
        fr_upper_bound = max_f - f_span * 0.01 if f_span > 0 else max_f * 1.01
        if fr_lower_bound >= fr_upper_bound or f_span == 0: # Handle narrow span or single point
            median_f = np.median(freqs) if len(freqs) > 0 else 1e9
            fr_lower_bound = median_f * 0.9
            fr_upper_bound = median_f * 1.1
        
        # Ensure initial fr is within bounds
        fr_init = np.clip(fr_init, fr_lower_bound, fr_upper_bound)

        # Bounds: fr, Qi, Qc, Qa. Qa can be large.
        # The model uses Qi/Qa, so Qa effectively acts as a magnitude.
        # For stability, Qa should not be extremely small (unless Qi is also small).
        bounds_lower = [fr_lower_bound, 10.0, 10.0, 1.0]      # Qa > 0
        bounds_upper = [fr_upper_bound, 1e9, 1e9, 1e10]   # Qa can be very large for symmetric case

        Qi_init = np.clip(Qi_init, bounds_lower[1], bounds_upper[1])
        Qc_init = np.clip(Qc_init, bounds_lower[2], bounds_upper[2])
        Qa_init = np.clip(np.abs(Qa_init) if Qa_init != 0 else bounds_lower[3] , bounds_lower[3], bounds_upper[3])
        
        params_init_vals = [fr_init, Qi_init, Qc_init, Qa_init]

        def residual_func_cpzm(params):
            fr_p, Qi_p, Qc_p, Qa_p = params
            # Basic check for non-physical Q values during iteration if issues persist
            if Qi_p <= 0 or Qc_p <=0 or Qa_p <=0: return np.full(2 * len(freqs), 1e6)

            model = cpzm_model(freqs, fr_p, Qi_p, Qc_p, Qa_p)
            res_complex = s21 - model

            if use_weights and self.weights is not None:
                w_sqrt = np.sqrt(self.weights)
                return np.hstack([res_complex.real * w_sqrt, res_complex.imag * w_sqrt])
            else:
                return np.hstack([res_complex.real, res_complex.imag])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) # For potential overflows/nans in model
                warnings.simplefilter("ignore", category=FutureWarning) # For scipy internal warnings
                ls_result = least_squares(
                    residual_func_cpzm, params_init_vals,
                    bounds=(bounds_lower, bounds_upper),
                    method='trf', ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=20000,
                    # loss='soft_l1', # Can try for noisy data
                    verbose=0 # Change to 1 or 2 for debugging
                )
        except Exception as e:
            print(f"CPZMFitter: least_squares optimization failed: {e}")
            self.fit_results = {
                'fr': fr_init, 'Qi': Qi_init, 'Qc': Qc_init, 'Qa': Qa_init, 'Ql': np.nan,
                'rmse': np.inf, 'error': str(e), 'weighted_fit': use_weights,
                'weights': self.weights, 'fr_estimate': self.fr_estimate,
                'ls_result': None, 'initial_params': initial_params
            }
            return self.fit_results


        fr_fit, Qi_fit, Qc_fit, Qa_fit = ls_result.x

        self.model_data = cpzm_model(freqs, fr_fit, Qi_fit, Qc_fit, Qa_fit)
        residuals = s21 - self.model_data
        rmse = np.sqrt(np.mean(np.abs(residuals)**2))

        # Calculate Ql from fitted parameters: 1/Ql = 1/Qi + 1/Qc (since Qc is real coupling Q here)
        Ql_fit_denom = (1.0/Qi_fit + 1.0/Qc_fit) if Qi_fit > 0 and Qc_fit > 0 else np.inf
        Ql_fit = 1.0 / Ql_fit_denom if Ql_fit_denom > 1e-18 else np.inf # Avoid division by zero
        
        self.fit_results = {
            'fr': fr_fit,
            'Qi': Qi_fit,
            'Qc': Qc_fit, 
            'Qa': Qa_fit, 
            'Ql': Ql_fit, 
            'rmse': rmse,
            'weighted_fit': use_weights,
            'weights': self.weights if use_weights else None,
            'fr_estimate': self.fr_estimate, # fr_estimate used for weights
            'ls_result_status': ls_result.status,
            'ls_result_message': ls_result.message,
            'ls_result_nfev': ls_result.nfev,
            'initial_params_used': params_init_vals # Store the actual initial values used by optimizer
        }
        if kwargs.get('return_ls_result_obj', False): # For debugging
            self.fit_results['ls_result_obj'] = ls_result
            
        return self.fit_results

    def get_model_data(self, freqs=None):
        """
        Return the CPZM model S21 data for given frequencies.
        """
        if not self.fit_results or 'fr' not in self.fit_results or np.isnan(self.fit_results['fr']): # Check if fit was successful
            # Fallback: if fit failed, try to return model with initial params if available
            if hasattr(self, 'initial_params_used_in_fit') and self.initial_params_used_in_fit:
                fr_i, Qi_i, Qc_i, Qa_i = self.initial_params_used_in_fit
                print("Warning: CPZM fit failed or no results. Returning model with initial parameters.")
                target_freqs = freqs if freqs is not None else self._intermediate_results['final_for_fitter'][0] if hasattr(self, '_intermediate_results') else None
                if target_freqs is None: raise ValueError("Cannot generate model data without freqs and failed fit.")
                return cpzm_model(target_freqs, fr_i, Qi_i, Qc_i, Qa_i)
            raise ValueError("Model not fitted successfully. Run fit() first.")

        target_freqs_internal = None
        if hasattr(self, '_intermediate_results') and 'final_for_fitter' in self._intermediate_results:
            target_freqs_internal = self._intermediate_results['final_for_fitter'][0]


        if freqs is None: # Use freqs from the data that was fit
            if self.model_data is not None:
                return self.model_data # Return the already computed model on original freqs
            elif target_freqs_internal is not None: # Recompute if model_data not stored but freqs available
                 freqs_to_use = target_freqs_internal
            else:
                 raise ValueError("Original frequencies for model generation not available and freqs argument is None.")
        else:
            freqs_to_use = freqs


        fr, Qi, Qc, Qa = self.fit_results['fr'], self.fit_results['Qi'], \
                         self.fit_results['Qc'], self.fit_results['Qa']
        return cpzm_model(freqs_to_use, fr, Qi, Qc, Qa)