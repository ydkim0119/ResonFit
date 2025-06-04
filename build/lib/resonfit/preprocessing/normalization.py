"""
Amplitude and phase normalization for resonator data.

This module provides the AmplitudePhaseNormalizer class for normalizing
S21 data such that the off-resonant point is at (1,0) in the complex plane.
"""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
import warnings

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
        radius, off-resonant point, phase fit parameters, etc.)
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

    def _fit_phase_model_internal(self, freqs_phase, phase_data, s21_mag_for_est=None, initial_params_phase=None):
        """
        Internal helper to fit phase data to the phase_model.
        Includes fallback to differential_evolution if curve_fit fails.
        """
        unwrapped_phase = np.unwrap(phase_data)
        min_f, max_f = (np.min(freqs_phase), np.max(freqs_phase)) if len(freqs_phase) > 0 else (0,0)
        f_span = max_f - min_f if len(freqs_phase) > 1 else 1.0

        # More robust bounds for fr
        fr_lower_bound = min_f + f_span * 0.01 if f_span > 0 else min_f * 0.99 if min_f > 0 else min_f - abs(min_f*0.01) - 1e3
        fr_upper_bound = max_f - f_span * 0.01 if f_span > 0 else max_f * 1.01 if max_f > 0 else max_f + abs(max_f*0.01) + 1e3
        if fr_lower_bound >= fr_upper_bound: # Handle very narrow span or single point
            median_f = np.median(freqs_phase) if len(freqs_phase) > 0 else 1e9
            fr_lower_bound = median_f * 0.9
            fr_upper_bound = median_f * 1.1

        lower_b = [fr_lower_bound, 10, -np.inf]
        upper_b = [fr_upper_bound, 1e8, np.inf]

        if initial_params_phase is None:
            if s21_mag_for_est is not None and len(s21_mag_for_est) > 0:
                idx_m = np.argmin(s21_mag_for_est)
            elif len(unwrapped_phase) > 1: # Use max gradient of phase
                idx_m = np.argmax(np.abs(np.gradient(unwrapped_phase)))
            else: # Fallback for single point or empty data
                idx_m = 0

            fr_e = freqs_phase[idx_m] if len(freqs_phase) > idx_m else np.mean(freqs_phase) if len(freqs_phase)>0 else 1e9
            fr_e = np.clip(fr_e, lower_b[0], upper_b[0])

            Ql_e = fr_e / f_span * 5.0 if f_span > 0 and fr_e != 0 else 1000.0
            if len(freqs_phase) > 10 and f_span > 0 and fr_e != 0: # Estimate Ql from phase slope
                dp_df = np.gradient(unwrapped_phase, freqs_phase)
                slope_at_res = dp_df[idx_m] if idx_m < len(dp_df) else np.mean(dp_df) if len(dp_df)>0 else 0
                if fr_e != 0 and slope_at_res !=0 : Ql_e = abs(slope_at_res * fr_e / (-4.0))

            Ql_e = np.clip(Ql_e, lower_b[1], upper_b[1])

            theta0_e_val = 0.0
            if len(unwrapped_phase) > 0:
                # Estimate theta0 from endpoints, avoiding resonance region
                num_edge_points = max(1, len(unwrapped_phase) // 10)
                edge_phases = np.concatenate((unwrapped_phase[:num_edge_points], unwrapped_phase[-num_edge_points:]))
                if len(edge_phases)>0: theta0_e_val = np.mean(edge_phases)
            initial_params_phase = (fr_e, Ql_e, theta0_e_val)
        else: # Clip provided initial params
            fr_e, Ql_e, theta0_e = initial_params_phase
            initial_params_phase = (
                np.clip(fr_e, lower_b[0], upper_b[0]),
                np.clip(Ql_e, lower_b[1], upper_b[1]),
                theta0_e
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=UserWarning) # For curve_fit covariance warning
                params, _ = curve_fit(
                    phase_model, freqs_phase, unwrapped_phase,
                    p0=initial_params_phase, bounds=(lower_b, upper_b), maxfev=5000
                )
            return params
        except (RuntimeError, ValueError) as e:
            print(f"Warning: Phase fitting with curve_fit failed: {e}. Trying differential_evolution.")
            try:
                def phase_objective(p_de):
                    model_de = phase_model(freqs_phase, p_de[0], p_de[1], p_de[2])
                    return np.sum((unwrapped_phase - model_de)**2)

                bounds_de = list(zip(lower_b, upper_b))
                # Widen bounds for theta0 slightly for DE if it's stuck
                bounds_de[2] = (initial_params_phase[2] - np.pi, initial_params_phase[2] + np.pi)

                res_de = differential_evolution(phase_objective, bounds_de, x0=initial_params_phase,
                                                popsize=15, tol=1e-5, mutation=(0.5,1.5), recombination=0.8, maxiter=300, polish=True)
                if res_de.success or res_de.fun < phase_objective(initial_params_phase): # Check if DE improved or succeeded
                    print(f"Alternative phase fitting successful: fr={res_de.x[0]/1e9:.6f} GHz, Ql={res_de.x[1]:.1f}, theta0={res_de.x[2]:.4f} rad")
                    return res_de.x
                else:
                     print("Alternative phase fitting did not improve. Using initial estimates from curve_fit failure point.")
            except Exception as e2:
                print(f"Alternative phase fitting (differential_evolution) error: {str(e2)}")

            print(f"Using initial estimates for phase (from before curve_fit): fr={initial_params_phase[0]/1e9:.6f} GHz, Ql={initial_params_phase[1]:.1f}, theta0={initial_params_phase[2]:.4f} rad")
            return initial_params_phase

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
        freqs_arr = np.asarray(freqs)
        s21_arr = np.asarray(s21)

        idx_min_abs_s21 = np.argmin(np.abs(s21_arr)) if len(s21_arr) > 0 else 0
        fr_estimate_for_weights = freqs_arr[idx_min_abs_s21] if len(freqs_arr) > 0 else np.mean(freqs_arr) if len(freqs_arr)>0 else 1e9

        circle_fit_weights = calculate_weights(freqs_arr, fr_estimate_for_weights, self.weight_bandwidth_scale)
        xc, yc, radius, circle_error = fit_circle_algebraic(s21_arr, circle_fit_weights)

        if np.isnan(xc) or np.isnan(yc) or np.isnan(radius):
            # Fallback if primary circle fit fails (e.g. data is a line)
            print("Warning: Primary circle fit failed in normalization. Attempting robust fallback.")
            # Try without weights or with different fr_estimate if necessary.
            # For simplicity, we'll use a very basic estimate if robust fit fails.
            xc = np.mean(s21_arr.real) if len(s21_arr) > 0 else 0
            yc = np.mean(s21_arr.imag) if len(s21_arr) > 0 else 0
            radius = np.max(np.abs(s21_arr - (xc + 1j*yc))) if len(s21_arr) > 0 else 1.0
            if radius == 0: radius = 1.0 # Avoid zero radius
            print(f"Fallback circle params: xc={xc:.3f}, yc={yc:.3f}, radius={radius:.3f}")

        s21_centered = s21_arr - (xc + 1j * yc)
        phase_centered_data = np.angle(s21_centered) # This is an array of phases for each freq point

        # Use the original fr_estimate_for_weights as a starting point for phase fit fr
        initial_phase_fr = fr_estimate_for_weights
        # Estimate Ql for phase fit based on FWHM of magnitude or default
        Ql_est_for_phase = initial_phase_fr / ((np.max(freqs_arr)-np.min(freqs_arr))/10.0) if (np.max(freqs_arr)-np.min(freqs_arr))>0 and initial_phase_fr !=0 else 1000.0

        phase_params = self._fit_phase_model_internal(
            freqs_arr, phase_centered_data,
            s21_mag_for_est=np.abs(s21_arr), # Pass |S21| for better initial fr_e in _fit_phase_model_internal
            initial_params_phase=(initial_phase_fr, Ql_est_for_phase, 0.0) # Initial theta0 as 0
        )
        fr_phase, Ql_phase, theta0_phase = phase_params

        beta_off_res = (theta0_phase + np.pi) % (2 * np.pi) # Off-resonance angle on circle
        P_off_resonant = (xc + 1j*yc) + radius * np.exp(1j*beta_off_res)

        a_norm_factor = np.abs(P_off_resonant)
        alpha_norm_offset = np.angle(P_off_resonant)

        if a_norm_factor == 0:
            print("Warning: Amplitude normalization factor 'a_norm_factor' is zero. Setting to 1.0 to avoid division by zero. Normalization may be incorrect.")
            a_norm_factor = 1.0

        s21_normalized = s21_arr / (a_norm_factor * np.exp(1j * alpha_norm_offset))

        self.normalization_params = {
            'xc_orig': xc, 'yc_orig': yc, 'radius_orig': radius, 'error_orig_circle_fit': circle_error,
            'weights_for_orig_circle_fit': circle_fit_weights,
            'fr_phase_fit': fr_phase, 'Ql_phase_fit': Ql_phase, 'theta0_phase_fit': theta0_phase,
            'phase_fit_input_freqs': freqs_arr.copy(), # Store a copy
            'phase_fit_input_centered_phase_data': phase_centered_data.copy(), # Store a copy
            'a_norm_factor': a_norm_factor, 'alpha_norm_offset': alpha_norm_offset,
            'P_off_resonant_orig_coords': P_off_resonant
        }

        return freqs_arr, s21_normalized

    def get_normalization_params(self):
        """
        Get the parameters determined during the normalization process.

        Returns
        -------
        dict
            A dictionary containing all stored normalization parameters.
        """
        if not self.normalization_params:
            raise ValueError("Normalization has not been performed yet. Run preprocess() first.")
        return self.normalization_params

    def __str__(self):
        """String representation with normalization status."""
        if self.normalization_params:
            a = self.normalization_params.get('a_norm_factor', 'unknown')
            alpha = self.normalization_params.get('alpha_norm_offset', 'unknown')
            status = f"a_norm={a:.3f}, alpha_norm={alpha:.3f} rad" if isinstance(a, float) and isinstance(alpha, float) else "params computed"
            return f"AmplitudePhaseNormalizer({status})"
        return "AmplitudePhaseNormalizer(not normalized)"