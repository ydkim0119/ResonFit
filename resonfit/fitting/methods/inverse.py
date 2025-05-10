"""
Inverse S21 Method fitting implementation.

This module provides the InverseFitter class which implements the Inverse S21 Method
for fitting resonator S21 data, fitting 1/S21 to a model that directly yields Qi.
This method is described in papers like:
A. Megrant et al., Appl. Phys. Lett. 100, 113510 (2012)
Keegan Mullins, "Highly Accurate Data Fitting for Microwave Resonators", Thesis (2021), Section 3.4
"""

import numpy as np
from scipy.optimize import least_squares
import warnings

from ...core.base import BaseFitter
from ...core.models import inverse_model # S21_inv = 1 + (Qi / (Qc_star * exp(1j*phi))) / (1 + 2j * Qi * (f - fr) / fr)
# Note: The inverse_model in core.models.py directly fits for Qi.
# The parameters are (f, fr, Qi, Qc_star, phi)
# Qc_star is related to the external Q but not identical to Qc_mag from DCM.

class InverseFitter(BaseFitter):
    """
    Fitter class implementing the Inverse S21 Method.

    This method fits the inverse of S21 data (1/S21) to directly extract
    the internal quality factor (Qi). The model for 1/S21 is typically simpler
    and can be more robust in certain noise conditions.
    """

    def __init__(self, weight_bandwidth_scale=1.0):
        """
        Initialize the InverseFitter.

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
        self.model_data_inv = None # Stores 1/S21_model
        self.model_data_s21 = None # Stores S21_model
        self.weights = None

    def _calculate_weights(self, freqs, s21_inv, fr_estimate=None, base_bandwidth_factor=0.1):
        """
        Calculate weights for fitting based on frequency distance from resonance.
        Note: For inverse S21, the resonance is a peak, not a dip.
        Weights should emphasize points around the peak of |1/S21|.

        Parameters
        ----------
        freqs : array_like
            Frequency array (Hz)
        s21_inv : array_like
            Complex INVERSE S21 data (1/S21)
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
            # Estimate resonance frequency from the peak of |1/S21|
            idx_res = np.argmax(np.abs(s21_inv)) if len(s21_inv) > 0 else 0
            fr_estimate = freqs[idx_res] if len(freqs) > idx_res else np.median(freqs)

        freq_diff = np.abs(freqs - fr_estimate)
        # For inverse S21, the "bandwidth" concept is related to the peak width.
        # We can use a similar approach for weighting.
        # Consider the frequency span where |1/S21| is significant.
        # Heuristic: Find points where |1/S21| drops to half its peak value.
        peak_val = np.abs(s21_inv[idx_res])
        half_peak_val = peak_val / np.sqrt(2) # for -3dB width of the peak power
        
        try:
            above_half_peak = np.where(np.abs(s21_inv) >= half_peak_val)[0]
            if len(above_half_peak) > 1:
                f_low = freqs[above_half_peak[0]]
                f_high = freqs[above_half_peak[-1]]
                bandwidth_est = f_high - f_low
                if bandwidth_est <=0: # fallback if something is wrong
                    bandwidth_est = base_bandwidth_factor * (np.max(freqs) - np.min(freqs)) if len(freqs) > 1 else 1.0
            else: # fallback
                bandwidth_est = base_bandwidth_factor * (np.max(freqs) - np.min(freqs)) if len(freqs) > 1 else 1.0
        except:
             bandwidth_est = base_bandwidth_factor * (np.max(freqs) - np.min(freqs)) if len(freqs) > 1 else 1.0


        final_bandwidth = bandwidth_est * self.weight_bandwidth_scale

        if final_bandwidth <= 0: # Ensure positive bandwidth
            weights = np.ones_like(freqs)
        else:
            weights = 1.0 / (1.0 + (freq_diff / final_bandwidth)**2)

        return weights, fr_estimate

    def _estimate_initial_params_inv(self, freqs, s21_inv):
        """Estimate initial parameters for the inverse S21 fit."""
        idx_peak = np.argmax(np.abs(s21_inv))
        fr_est = freqs[idx_peak]

        # Estimate Qi from bandwidth of 1/|S21|^2 peak (approximate Lorentzian)
        s21_inv_mag_sq = np.abs(s21_inv)**2
        peak_mag_sq = s21_inv_mag_sq[idx_peak]
        half_max_mag_sq = peak_mag_sq / 2.0

        try:
            # Find points where magnitude squared is half max
            left_half_idx = np.where((freqs < fr_est) & (s21_inv_mag_sq >= half_max_mag_sq))[0]
            right_half_idx = np.where((freqs > fr_est) & (s21_inv_mag_sq >= half_max_mag_sq))[0]
            
            f1 = freqs[left_half_idx[-1]] if len(left_half_idx) > 0 else freqs[0]
            f2 = freqs[right_half_idx[0]] if len(right_half_idx) > 0 else freqs[-1]
            
            bandwidth_est = f2 - f1
            if bandwidth_est <= 0: # fallback
                 bandwidth_est = (np.max(freqs)-np.min(freqs))/20.0
            Qi_est = fr_est / bandwidth_est if bandwidth_est > 0 else 10000.0
        except:
            bandwidth_est = (np.max(freqs)-np.min(freqs))/20.0
            Qi_est = fr_est / bandwidth_est if bandwidth_est > 0 else 10000.0


        # Estimate Qc_star and phi
        # At resonance, 1/S21_res = 1 + Qi / (Qc_star * exp(1j*phi))
        # Let val_at_res = s21_inv[idx_peak]
        # (val_at_res - 1) = Qi / (Qc_star * exp(1j*phi))
        # Qc_star * exp(1j*phi) = Qi / (val_at_res - 1)
        val_at_res_minus_1 = s21_inv[idx_peak] - 1.0
        if np.abs(val_at_res_minus_1) < 1e-9: # Avoid division by zero if s21_inv[idx_peak] is close to 1
            Qc_star_est = Qi_est # Default to Qi ~ Qc_star
            phi_est = 0.0
        else:
            Qc_star_exp_jphi = Qi_est / val_at_res_minus_1
            Qc_star_est = np.abs(Qc_star_exp_jphi)
            phi_est = np.angle(Qc_star_exp_jphi)
        
        # Ensure phi is in a reasonable range, e.g., -pi/2 to pi/2 for typical transmission resonators
        phi_est = np.unwrap([phi_est])[0] # unwrap to handle jumps
        phi_est = np.mod(phi_est + np.pi/2, np.pi) - np.pi/2 # map to [-pi/2, pi/2] range if needed

        return {
            'fr': fr_est,
            'Qi': max(10.0, Qi_est), # Ensure positive Q
            'Qc_star': max(10.0, Qc_star_est), # Ensure positive Q
            'phi': phi_est
        }

    def fit(self, freqs, s21, use_weights=True, initial_params=None, **kwargs):
        """
        Fit the resonator data using the Inverse S21 method.

        Parameters
        ----------
        freqs : array_like
            Frequency array (Hz)
        s21 : array_like
            Complex S21 data (original, not inverted)
        use_weights : bool, optional
            Whether to use frequency-dependent weighting, by default True
        initial_params : dict, optional
            Initial parameters for fitting:
            - fr: resonance frequency (Hz)
            - Qi: internal quality factor
            - Qc_star: modified coupling quality factor magnitude
            - phi: impedance mismatch angle (radians)
            If None, values will be estimated from data.
        **kwargs : dict
            Additional parameters for weighting or initial estimation.

        Returns
        -------
        dict
            Fitting results containing keys:
            - fr: resonance frequency (Hz)
            - Qi: internal quality factor
            - Qc_star: modified coupling quality factor
            - phi: impedance mismatch angle (radians)
            - Ql: loaded quality factor (calculated from Qi and Qc_star)
            - rmse: root mean square error of the 1/S21 fit
            - weighted_fit: whether weighting was used
            - weights: weights array if weighting was used
        """
        if len(freqs) != len(s21):
            raise ValueError("freqs and s21 must have the same length")
        if len(freqs) < 5: # Need enough points for fitting
            raise ValueError("Not enough data points for fitting (minimum 5)")

        s21_inv = 1.0 / s21 # Invert the data

        base_bandwidth_factor = kwargs.get('base_bandwidth_factor', 0.1)
        fr_est_input = kwargs.get('fr_estimate', None)

        self.weights = None
        if use_weights:
            self.weights, self.fr_estimate = self._calculate_weights(
                freqs, s21_inv, fr_estimate=fr_est_input,
                base_bandwidth_factor=base_bandwidth_factor
            )
        else:
            if fr_est_input is not None:
                self.fr_estimate = fr_est_input
            else:
                idx_res = np.argmax(np.abs(s21_inv)) if len(s21_inv) > 0 else 0
                self.fr_estimate = freqs[idx_res] if len(freqs) > idx_res else np.median(freqs)


        if initial_params is None:
            initial_params = self._estimate_initial_params_inv(freqs, s21_inv)
            if self.fr_estimate : initial_params['fr'] = self.fr_estimate # Use fr_estimate from weighting if available

        required_params = ['fr', 'Qi', 'Qc_star', 'phi']
        missing_params = [p for p in required_params if p not in initial_params]
        if missing_params:
            raise ValueError(f"Missing required initial parameters for InverseFitter: {missing_params}")

        fr_init = initial_params['fr']
        Qi_init = initial_params['Qi']
        Qc_star_init = initial_params['Qc_star']
        phi_init = initial_params['phi']

        min_f, max_f = np.min(freqs), np.max(freqs)
        f_span = max_f - min_f
        
        fr_lower = min_f + f_span * 0.01
        fr_upper = max_f - f_span * 0.01
        if fr_lower >= fr_upper : # Handle narrow span or single point
            median_f = np.median(freqs)
            fr_lower = median_f * 0.99
            fr_upper = median_f * 1.01

        bounds_lower = [fr_lower, 10.0, 10.0, -np.pi] # Looser phi bounds initially
        bounds_upper = [fr_upper, 1e9, 1e9, np.pi]   # Looser phi bounds initially

        fr_init = np.clip(fr_init, bounds_lower[0], bounds_upper[0])
        Qi_init = np.clip(Qi_init, bounds_lower[1], bounds_upper[1])
        Qc_star_init = np.clip(Qc_star_init, bounds_lower[2], bounds_upper[2])
        phi_init = np.clip(phi_init, bounds_lower[3], bounds_upper[3])

        params_init = [fr_init, Qi_init, Qc_star_init, phi_init]

        def residual_func_inv(params):
            fr, Qi, Qc_star, phi = params
            model_inv = inverse_model(freqs, fr, Qi, Qc_star, phi)
            res_complex = s21_inv - model_inv

            if use_weights and self.weights is not None:
                w_sqrt = np.sqrt(self.weights)
                return np.hstack([res_complex.real * w_sqrt, res_complex.imag * w_sqrt])
            else:
                return np.hstack([res_complex.real, res_complex.imag])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore overflow/invalid value in S21_inv if s21 is zero
            ls_result = least_squares(
                residual_func_inv, params_init,
                bounds=(bounds_lower, bounds_upper),
                method='trf', ftol=1e-9, xtol=1e-9, max_nfev=10000
            )

        fr_fit, Qi_fit, Qc_star_fit, phi_fit = ls_result.x
        
        # Ensure phi is in [-pi/2, pi/2] for consistency with DCM if desired, or [-pi, pi]
        # phi_fit = np.arctan2(np.sin(phi_fit), np.cos(phi_fit)) # Normalize to [-pi, pi]
        
        self.model_data_inv = inverse_model(freqs, fr_fit, Qi_fit, Qc_star_fit, phi_fit)
        self.model_data_s21 = 1.0 / self.model_data_inv

        residuals_inv = s21_inv - self.model_data_inv
        rmse_inv = np.sqrt(np.mean(np.abs(residuals_inv)**2))

        # Calculate Ql for completeness, using the definition 1/Ql = 1/Qi + Re(exp(1j*phi)/Qc_star)
        # This is based on the relation S21 = 1 - (Ql/Qi) * (Qi * exp(-1j*phi) / Qc_star) / (1 + 2jQl (f-fr)/fr)
        # After some algebra, and comparing to S21 = 1 - (Ql/Qc_complex)/(1+2jQl df/f)
        # where Qc_complex = Qc_mag * exp(-1j*phi_dcm)
        # We can relate Qc_star, phi (INV) to Qc_mag, phi (DCM) via Qi/Qc_complex = Qi * exp(1j*phi_inv) / Qc_star_inv
        # So, Qc_complex = Qc_star_inv * exp(-1j*phi_inv)
        # Then 1/Ql = 1/Qi + Re(1/Qc_complex) = 1/Qi + Re(exp(1j*phi_inv)/Qc_star_inv)
        # Ql can also be found from: 1/Ql = 1/Qi + Re(exp(1j*phi_fit)/Qc_star_fit)
        # Or, from thesis (Keegan Mullins, section 3.4)
        # Ql is derived from the definition S21 = 1 - (Ql/Qc_eff) / (1 + 2j Ql (f-fr)/fr)
        # where 1/S21 = 1 + (Ql/Qi) / (1 + 2j Ql (f-fr)/fr) if Qc_eff = Qi
        # More generally, S_21 = 1 - \frac{Q_l/Q_c^* e^{-i\phi}}{1+2iQ_l(f-f_r)/f_r}
        # Then 1/S_21 = \frac{1+2iQ_l(f-f_r)/f_r}{1 - Q_l/Q_c^* e^{-i\phi} + 2iQ_l(f-f_r)/f_r}
        # This doesn't directly match the inverse_model form.
        # The model used: S21_inv = 1 + (Qi / (Qc_star * exp(1j*phi))) / (1 + 2j * Qi * (f - fr) / fr)
        # This implies that at resonance (f=fr), S21_inv_res = 1 + Qi / (Qc_star * exp(1j*phi))
        # For a standard S21 model: S21 = 1 - (Ql/Qc_mag * exp(-j phi_dcm)) / (1 + 2j Ql (f-fr)/fr)
        # S21_res = 1 - Ql/(Qc_mag * exp(-j phi_dcm))
        # Let Q_ext_complex = Qc_star * exp(j phi_fit). Then S21_inv_res = 1 + Qi / Q_ext_complex
        # S21_res = Q_ext_complex / (Q_ext_complex + Qi)
        # Comparing coefficients, this means 1/Ql = 1/Qi + Re(1/Q_ext_complex) = 1/Qi + Re(exp(-j phi_fit)/Qc_star)
        # So Ql_inv_calc = 1.0 / Qi_fit + np.real(np.exp(-1j * phi_fit) / Qc_star_fit)
        
        # From Probst et al. (Rev. Sci. Instrum. 86, 024706 (2015)), Eq (10) & (11)
        # S21 = (1 - Ql/Qc_int exp(-2j delta_0)) / (1 - Ql/Qc_ext exp(-j phi) exp(-2j delta_0) + 2j Ql x)
        # This is more complex.
        # Let's use the simpler relation derived by equating the form of S21 from DCM and from INV's parameters.
        # S21_DCM_res = 1 - Ql_dcm / (Qc_mag_dcm * exp(-j*phi_dcm))
        # S21_INV_res = (Qc_star_fit * exp(j*phi_fit)) / (Qc_star_fit * exp(j*phi_fit) + Qi_fit)
        # If S21_DCM_res = S21_INV_res, and assuming Ql_dcm = Ql_inv
        # And Qc_mag_dcm * exp(-j*phi_dcm)  = Qc_star_fit * exp(j*phi_fit)
        # Then 1/Ql = 1/Qi_fit + Re(1 / (Qc_star_fit * exp(j*phi_fit)))
        # 1/Ql = 1/Qi_fit + np.cos(phi_fit) / Qc_star_fit
        # (This is from Gao thesis, relating Ql, Qi, Qc, phi for S21 model when Qc is complex)
        # Ql = 1.0 / (1.0/Qi_fit + np.cos(phi_fit)/Qc_star_fit) if (1.0/Qi_fit + np.cos(phi_fit)/Qc_star_fit) > 0 else np.inf
        
        # The model S21_inv = 1 + (Qi / (Qc_star * exp(1j*phi))) / (1 + 2j * Qi * (f - fr) / fr)
        # implies Qc = Qc_star / exp(1j*phi) and Ql = Qi. This is not general.
        #
        # The more standard INV model from literature like Megrant or Mullins thesis (eq 3.5) is:
        # 1/S21 = 1 + (Ql/Qi) / (1 + 2i Ql (f-fr)/fr) * (Qc_star * exp(i phi) / Ql)
        # No, that's not it. Mullins eq 3.5: S21^-1 = 1 + (Qi / (Qc* exp(i phi))) / (1 + 2i Qi (f-fr)/fr)
        # This directly gives Qi. To get Ql, one needs to define Qc from Qc* and phi.
        # If Qc_complex = Qc_star * exp(i phi) is the effective external Q in the inverted model sense,
        # then Ql = 1 / (1/Qi + 1/abs(Qc_star)) is often used as an approximation if phi is small.
        # Or using 1/Ql = 1/Qi + Re(1/Qc_complex)
        Ql_calc_denom = (1.0/Qi_fit + np.cos(phi_fit)/Qc_star_fit) # Assuming Qc_complex for S21 form is Qc_star*exp(j*phi)
        Ql_fit = 1.0 / Ql_calc_denom if Ql_calc_denom > 1e-12 else np.inf


        self.fit_results = {
            'fr': fr_fit,
            'Qi': Qi_fit,
            'Qc_star': Qc_star_fit, # This is the Q_c^* from the inverse model
            'phi': phi_fit,         # This is phi from the inverse model
            'Ql': Ql_fit,           # Calculated Ql
            'rmse_inv_S21': rmse_inv, # RMSE of the 1/S21 fit
            'weighted_fit': use_weights,
            'weights': self.weights if use_weights else None,
            'fr_estimate': self.fr_estimate,
            'ls_result': ls_result,
            'initial_params': initial_params,
            's21_inv_data_for_fit': s21_inv # Store the inverted data used for fitting
        }

        return self.fit_results

    def get_model_data(self, freqs=None):
        """
        Return the S21 model data for given frequencies using fitted parameters.
        Note: This returns S21, not 1/S21.

        Parameters
        ----------
        freqs : array_like, optional
            Frequency array (Hz). If None, uses the frequencies from the fit.

        Returns
        -------
        array_like
            Complex S21 model data
        """
        if not self.fit_results:
            raise ValueError("Model not fitted yet. Run fit() first.")

        if freqs is None:
            if self.model_data_s21 is None: # Should have been computed in fit()
                 fr = self.fit_results['fr']
                 Qi = self.fit_results['Qi']
                 Qc_star = self.fit_results['Qc_star']
                 phi = self.fit_results['phi']
                 # This should have been called with original freqs from fit
                 # Need to store original freqs or require freqs argument here.
                 # For now, assume self.model_data_s21 is populated.
                 raise ValueError("Original frequencies for model generation not available and freqs argument is None.")
            return self.model_data_s21

        fr = self.fit_results['fr']
        Qi = self.fit_results['Qi']
        Qc_star = self.fit_results['Qc_star']
        phi = self.fit_results['phi']

        model_inv_custom_freqs = inverse_model(freqs, fr, Qi, Qc_star, phi)
        return 1.0 / model_inv_custom_freqs

    def get_inverse_model_data(self, freqs=None):
        """
        Return the 1/S21 model data for given frequencies.

        Parameters
        ----------
        freqs : array_like, optional
            Frequency array (Hz). If None, uses the frequencies from the fit.

        Returns
        -------
        array_like
            Complex 1/S21 model data
        """
        if not self.fit_results:
            raise ValueError("Model not fitted yet. Run fit() first.")

        if freqs is None:
            return self.model_data_inv # This was calculated with original freqs

        fr = self.fit_results['fr']
        Qi = self.fit_results['Qi']
        Qc_star = self.fit_results['Qc_star']
        phi = self.fit_results['phi']
        return inverse_model(freqs, fr, Qi, Qc_star, phi)