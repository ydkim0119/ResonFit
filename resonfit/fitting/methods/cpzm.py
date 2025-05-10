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
        Weights emphasize points around the resonance dip.
        """
        if fr_estimate is None:
            # Estimate resonance frequency from the dip of |S21|
            idx_res = np.argmin(np.abs(s21)) if len(s21) > 0 else 0
            fr_estimate = freqs[idx_res] if len(freqs) > idx_res else np.median(freqs)

        freq_diff = np.abs(freqs - fr_estimate)
        
        # Estimate bandwidth based on FWHM of the dip
        s21_mag = np.abs(s21)
        min_mag = s21_mag[idx_res]
        # For CPZM, the off-resonant magnitude is not necessarily 1 if not perfectly normalized for this model.
        # However, for weighting, we estimate bandwidth around the dip.
        # Use a heuristic or assume it's normalized enough that max_mag is around 1.
        max_mag_far_from_res = np.max([s21_mag[0], s21_mag[-1], np.median(s21_mag)]) if len(s21_mag) > 2 else 1.0
        
        # Target magnitude for FWHM: min_mag + (max_mag_far_from_res - min_mag)/2
        # Or, more simply, use a fraction of the frequency span if Ql is unknown.
        try:
            # Find -3dB points from the minimum transmission (dip)
            s21_db = 20 * np.log10(s21_mag)
            min_db = s21_db[idx_res]
            target_db = min_db + 3.0
            
            left_idx = np.where((freqs < fr_estimate) & (s21_db >= target_db))[0]
            right_idx = np.where((freqs > fr_estimate) & (s21_db >= target_db))[0]

            f_low = freqs[left_idx[-1]] if len(left_idx) > 0 else freqs[0]
            f_high = freqs[right_idx[0]] if len(right_idx) > 0 else freqs[-1]
            bandwidth_est = f_high - f_low
            if bandwidth_est <= 0:
                 bandwidth_est = base_bandwidth_factor * (np.max(freqs) - np.min(freqs)) if len(freqs) > 1 else 1.0
        except:
            bandwidth_est = base_bandwidth_factor * (np.max(freqs) - np.min(freqs)) if len(freqs) > 1 else 1.0


        final_bandwidth = bandwidth_est * self.weight_bandwidth_scale

        if final_bandwidth <= 0:
            weights = np.ones_like(freqs)
        else:
            weights = 1.0 / (1.0 + (freq_diff / final_bandwidth)**2)
        return weights, fr_estimate

    def _estimate_initial_params_cpzm(self, freqs, s21):
        """Estimate initial parameters for the CPZM fit."""
        idx_min = np.argmin(np.abs(s21))
        fr_est = freqs[idx_min]

        # Estimate Ql (loaded Q) from FWHM
        s21_mag = np.abs(s21)
        min_val = s21_mag[idx_min]
        
        # Find approx off-resonant magnitude (e.g. average of first/last 10% of points)
        num_edge_points = max(1, len(freqs) // 10)
        off_res_mag_est = np.mean(np.concatenate((s21_mag[:num_edge_points], s21_mag[-num_edge_points:]))) if len(freqs) > 2*num_edge_points else 1.0
        
        half_power_level = np.sqrt( (min_val**2 + off_res_mag_est**2) / 2.0 ) # RMS average for magnitude
        
        try:
            left_half_idx = np.where((freqs < fr_est) & (s21_mag >= half_power_level))[0]
            right_half_idx = np.where((freqs > fr_est) & (s21_mag >= half_power_level))[0]
            f1 = freqs[left_half_idx[-1]] if len(left_half_idx) > 0 else freqs[0]
            f2 = freqs[right_half_idx[0]] if len(right_half_idx) > 0 else freqs[-1]
            bandwidth_est = f2 - f1
            if bandwidth_est <= 0: bandwidth_est = (np.max(freqs)-np.min(freqs))/10.0
            Ql_est = fr_est / bandwidth_est if bandwidth_est > 0 else 10000.0
        except:
            bandwidth_est = (np.max(freqs)-np.min(freqs))/10.0
            Ql_est = fr_est / bandwidth_est if bandwidth_est > 0 else 10000.0

        # For CPZM, Qi is fit directly. Qc and Qa are also fit.
        # Initial guess: Qi ~ Ql, Qc ~ Ql (or 2*Ql if undercoupled). Qa can be large (symmetric) or smaller.
        Qi_est = Ql_est
        Qc_est = Ql_est * (off_res_mag_est / min_val) if min_val > 1e-9 else Ql_est * 2.0 # Rough relation based on coupling depth
        
        # Estimate Qa from circle rotation.
        # If data is normalized (off-res near (1,0)), then s21_res ~ (1 - Qi/Qc - j Qi/Qa) / (1 + Qi/Qc + j Qi/Qa)
        # Phase of s21_res can give hint about Qa.
        # More simply, start Qa large (e.g., 10*Qi) for symmetric guess.
        Qa_est = 10 * Qi_est # Start with a relatively symmetric line shape.

        return {
            'fr': fr_est,
            'Qi': max(10.0, Qi_est),
            'Qc': max(10.0, Qc_est),
            'Qa': max(10.0, Qa_est) # Qa can be positive or negative in some conventions, but model uses magnitude.
                                    # For this model, Qa is positive.
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

        base_bandwidth_factor = kwargs.get('base_bandwidth_factor', 0.1)
        fr_est_input = kwargs.get('fr_estimate', None)

        self.weights = None
        if use_weights:
            self.weights, self.fr_estimate = self._calculate_weights(
                freqs, s21, fr_estimate=fr_est_input,
                base_bandwidth_factor=base_bandwidth_factor
            )
        else:
            if fr_est_input is not None:
                self.fr_estimate = fr_est_input
            else:
                idx_res = np.argmin(np.abs(s21)) if len(s21) > 0 else 0
                self.fr_estimate = freqs[idx_res] if len(freqs) > idx_res else np.median(freqs)


        if initial_params is None:
            initial_params = self._estimate_initial_params_cpzm(freqs, s21)
            if self.fr_estimate : initial_params['fr'] = self.fr_estimate


        required_params = ['fr', 'Qi', 'Qc', 'Qa']
        missing_params = [p for p in required_params if p not in initial_params]
        if missing_params:
            raise ValueError(f"Missing required initial parameters for CPZMFitter: {missing_params}")

        fr_init, Qi_init, Qc_init, Qa_init = initial_params['fr'], initial_params['Qi'], \
                                             initial_params['Qc'], initial_params['Qa']

        min_f, max_f = np.min(freqs), np.max(freqs)
        f_span = max_f - min_f
        
        fr_lower = min_f + f_span * 0.01
        fr_upper = max_f - f_span * 0.01
        if fr_lower >= fr_upper :
            median_f = np.median(freqs)
            fr_lower = median_f * 0.99
            fr_upper = median_f * 1.01


        # Bounds: fr, Qi, Qc, Qa. Qa can be large.
        # The sign of Qa determines rotation direction in some conventions, but model uses magnitude implicitly.
        # Here, the model takes Qi/Qa, so Qa effectively acts as a magnitude.
        bounds_lower = [fr_lower, 10.0, 10.0, 1.0]      # Qa > 0
        bounds_upper = [fr_upper, 1e9, 1e9, 1e10]   # Qa can be very large for symmetric case

        fr_init = np.clip(fr_init, bounds_lower[0], bounds_upper[0])
        Qi_init = np.clip(Qi_init, bounds_lower[1], bounds_upper[1])
        Qc_init = np.clip(Qc_init, bounds_lower[2], bounds_upper[2])
        Qa_init = np.clip(Qa_init, bounds_lower[3], bounds_upper[3])
        
        # If Qa_init is negative from some estimation, take abs, as model implies magnitude.
        Qa_init = np.abs(Qa_init) if Qa_init != 0 else bounds_lower[3]


        params_init = [fr_init, Qi_init, Qc_init, Qa_init]

        def residual_func_cpzm(params):
            fr, Qi, Qc, Qa = params
            model = cpzm_model(freqs, fr, Qi, Qc, Qa)
            res_complex = s21 - model

            if use_weights and self.weights is not None:
                w_sqrt = np.sqrt(self.weights)
                return np.hstack([res_complex.real * w_sqrt, res_complex.imag * w_sqrt])
            else:
                return np.hstack([res_complex.real, res_complex.imag])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ls_result = least_squares(
                residual_func_cpzm, params_init,
                bounds=(bounds_lower, bounds_upper),
                method='trf', ftol=1e-9, xtol=1e-9, max_nfev=10000
            )

        fr_fit, Qi_fit, Qc_fit, Qa_fit = ls_result.x

        self.model_data = cpzm_model(freqs, fr_fit, Qi_fit, Qc_fit, Qa_fit)
        residuals = s21 - self.model_data
        rmse = np.sqrt(np.mean(np.abs(residuals)**2))

        # Calculate Ql from fitted parameters: 1/Ql = 1/Qi + 1/Qc (since Qc is real coupling Q here)
        Ql_fit_denom = (1.0/Qi_fit + 1.0/Qc_fit)
        Ql_fit = 1.0 / Ql_fit_denom if Ql_fit_denom > 1e-12 else np.inf
        
        # phi can be related to Qa by tan(phi) = -Qi/Qa (Deng et al. or Gao thesis conventions)
        # phi_eff = np.arctan(-Qi_fit / Qa_fit)
        # Note: this phi is an effective phase rotation of the resonance circle, different from DCM's phi.
        # For CPZM, the primary output is Qi, Qc, Qa.

        self.fit_results = {
            'fr': fr_fit,
            'Qi': Qi_fit,
            'Qc': Qc_fit, # Real coupling Q
            'Qa': Qa_fit, # Asymmetry Q
            'Ql': Ql_fit, # Loaded Q calculated from Qi, Qc
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
        Return the CPZM model S21 data for given frequencies.
        """
        if not self.fit_results:
            raise ValueError("Model not fitted yet. Run fit() first.")

        if freqs is None:
            return self.model_data

        fr, Qi, Qc, Qa = self.fit_results['fr'], self.fit_results['Qi'], \
                         self.fit_results['Qc'], self.fit_results['Qa']
        return cpzm_model(freqs, fr, Qi, Qc, Qa)