"""
Plotting utilities for resonator data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from ..core.models import phase_model # For plotting phase model


class ResonancePlotter:
    """
    Class for plotting resonator data and fitting results.
    
    Provides visualization methods for various stages of the resonator
    fitting process, including raw data, preprocessing steps, and fitting results.
    """
    
    def __init__(self, figsize=(12, 9), style='default'):
        """
        Initialize the ResonancePlotter with style settings.
        
        Parameters
        ----------
        figsize : tuple, optional
            Default figure size for plots, by default (12, 9) for 2x2, (14,6) for 1x2
        style : str, optional
            Matplotlib style to use, by default 'default'
        """
        self.figsize_2x2 = figsize
        self.figsize_1x2 = (14,6)
        self.figsize_single = (10,5)
        self.figsize_2x1_shared = (10,8)


        # Set style if provided
        if style != 'default':
            try:
                plt.style.use(style)
            except:
                print(f"Warning: Matplotlib style '{style}' not found. Using default.")
            
        # Default plot style parameters
        self.plot_styles = {
            'original_data_color': 'blue',
            'delay_corrected_color': 'red',
            'normalized_data_color': 'purple',
            'inverse_data_color': 'teal', # For 1/S21 data
            'model_fit_color': 'black', # Generic model fit line (e.g. circle)
            's21_model_color': 'green', # S21 model from fit
            's21_inv_model_color': 'darkorange', # 1/S21 model from fit
            'phase_model_color': 'orangered',
            'point_alpha': 0.5,
            'line_linewidth': 1.0,
            'fit_linewidth': 1.5, # Circle fit lines
            'model_linewidth': 2.0, # S21 model lines
            'marker_size_base': 20, # Base marker size for scatter
            'marker_size_min_weighted': 5, # Min marker size for weighted scatter
            'marker_size_scale_weighted': 50, # Scale factor for weighted scatter
            'marker_style': 'o',
            'scatter_cmap_weights': 'viridis_r', 
            'scatter_cmap_residuals': 'coolwarm',
            'axis_label_fontsize': 10,
            'title_fontsize': 12,
            'legend_fontsize': 8,
            'grid_alpha': 0.3,
            'accent_color_1': 'darkgreen', # For special points like P_off, resonance
            'accent_color_2': 'black', # For centers, origin
            'vline_color': 'dimgray',
        }
    
    def plot_raw_data(self, freqs, s21, title="Raw S21 Data"):
        """
        Plot the raw S21 data in magnitude/phase format and complex plane.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_2x2)
        fig.suptitle(title, fontsize=self.plot_styles['title_fontsize'] + 2)
        ps = self.plot_styles
        
        axes[0, 0].plot(freqs / 1e9, 20 * np.log10(np.abs(s21)), 
                      color=ps['original_data_color'], linewidth=ps['line_linewidth'])
        axes[0, 0].set_xlabel('Frequency (GHz)', fontsize=ps['axis_label_fontsize'])
        axes[0, 0].set_ylabel('|S21| (dB)', fontsize=ps['axis_label_fontsize'])
        axes[0, 0].set_title('Magnitude', fontsize=ps['title_fontsize'])
        axes[0, 0].grid(True, alpha=ps['grid_alpha'])
        
        axes[0, 1].plot(freqs / 1e9, np.unwrap(np.angle(s21)),
                      color=ps['original_data_color'], linewidth=ps['line_linewidth'])
        axes[0, 1].set_xlabel('Frequency (GHz)', fontsize=ps['axis_label_fontsize'])
        axes[0, 1].set_ylabel('Phase (rad)', fontsize=ps['axis_label_fontsize'])
        axes[0, 1].set_title('Unwrapped Phase', fontsize=ps['title_fontsize'])
        axes[0, 1].grid(True, alpha=ps['grid_alpha'])
        
        axes[1, 0].plot(s21.real, s21.imag, marker=ps['marker_style'], ls='None',
                       color=ps['original_data_color'], alpha=ps['point_alpha'], markersize=np.sqrt(ps['marker_size_base']))
        axes[1, 0].set_xlabel('Real', fontsize=ps['axis_label_fontsize'])
        axes[1, 0].set_ylabel('Imaginary', fontsize=ps['axis_label_fontsize'])
        axes[1, 0].set_title('Complex Plane', fontsize=ps['title_fontsize'])
        axes[1, 0].grid(True, alpha=ps['grid_alpha']); axes[1, 0].axis('equal')
        
        axes[1, 1].plot(freqs / 1e9, np.angle(s21), # Not unwrapped for this view
                      color=ps['original_data_color'], linewidth=ps['line_linewidth'])
        axes[1, 1].set_xlabel('Frequency (GHz)', fontsize=ps['axis_label_fontsize'])
        axes[1, 1].set_ylabel('Phase (rad)', fontsize=ps['axis_label_fontsize'])
        axes[1, 1].set_title('Phase (Wrapped)', fontsize=ps['title_fontsize'])
        axes[1, 1].grid(True, alpha=ps['grid_alpha'])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        return fig
    
    def plot_delay_correction(self, freqs, s21_original, s21_corrected, delay_ns, 
                              corrected_circle_params=None, weights=None):
        """
        Plot original and delay-corrected S21 data in the complex plane.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize_1x2)
        ps = self.plot_styles
        
        axes[0].plot(s21_original.real, s21_original.imag, marker=ps['marker_style'], ls='None',
                     color=ps['original_data_color'], alpha=ps['point_alpha'], 
                     markersize=np.sqrt(ps['marker_size_base']), label='Original Data')
        axes[0].set_title('Before Delay Correction', fontsize=ps['title_fontsize'])
        axes[0].set_xlabel('Real', fontsize=ps['axis_label_fontsize'])
        axes[0].set_ylabel('Imaginary', fontsize=ps['axis_label_fontsize'])
        axes[0].grid(True, alpha=ps['grid_alpha']); axes[0].axis('equal')
        axes[0].legend(fontsize=ps['legend_fontsize'])
        
        point_sizes_corrected = ps['marker_size_base']
        if weights is not None and len(weights) == len(s21_corrected) and np.max(weights) > 0:
            weights_norm = weights / np.max(weights)
            point_sizes_corrected = ps['marker_size_min_weighted'] + ps['marker_size_scale_weighted'] * weights_norm
            data_label_corrected = 'Delay Corrected (size by weight)'
        else:
            data_label_corrected = 'Delay Corrected Data'

        axes[1].scatter(s21_corrected.real, s21_corrected.imag, s=point_sizes_corrected,
                        color=ps['delay_corrected_color'], alpha=ps['point_alpha'], label=data_label_corrected)
        
        if corrected_circle_params and all(k in corrected_circle_params for k in ['xc', 'yc', 'radius']):
            xc, yc, r = corrected_circle_params['xc'], corrected_circle_params['yc'], corrected_circle_params['radius']
            if not (np.isnan(xc) or np.isnan(yc) or np.isnan(r) or r == 0):
                theta_circle = np.linspace(0, 2 * np.pi, 200)
                circle_x = xc + r * np.cos(theta_circle)
                circle_y = yc + r * np.sin(theta_circle)
                axes[1].plot(circle_x, circle_y, color=ps['model_fit_color'],
                             lw=ps['fit_linewidth'], label='Fitted Circle (Corrected)')
                axes[1].plot(xc, yc, marker='x', color=ps['accent_color_2'],
                             ms=np.sqrt(ps['marker_size_base'])*1.5, label='Circle Center')
            
        axes[1].plot(0, 0, marker='+', color=ps['accent_color_2'], ms=np.sqrt(ps['marker_size_base'])*1.5, label='Origin')
        axes[1].set_title(f'After Delay Correction (delay = {delay_ns:.3f} ns)', fontsize=ps['title_fontsize'])
        axes[1].set_xlabel('Real', fontsize=ps['axis_label_fontsize'])
        axes[1].set_ylabel('Imaginary', fontsize=ps['axis_label_fontsize'])
        axes[1].grid(True, alpha=ps['grid_alpha']); axes[1].axis('equal')
        axes[1].legend(fontsize=ps['legend_fontsize'])
        
        plt.tight_layout()
        return fig

    def plot_delay_correction_magnitude_vs_freq(self, freqs, s21_original, s21_corrected, 
                                                weights=None, fr_estimate_for_weights=None, title_suffix=""):
        """Plots S21 magnitude original vs delay corrected, with weighted points."""
        fig, ax = plt.subplots(figsize=self.figsize_single)
        ps = self.plot_styles

        ax.plot(freqs / 1e9, 20 * np.log10(np.abs(s21_original)), color=ps['original_data_color'],
                 ls='-', alpha=ps['point_alpha'], lw=ps['line_linewidth'], label='Original Mag')
        
        point_sizes_corrected = ps['marker_size_base']
        label_corrected = 'Delay Corrected Mag'
        if weights is not None and len(weights) == len(s21_corrected) and np.max(weights) > 0:
            weights_norm = weights / np.max(weights)
            point_sizes_corrected = ps['marker_size_min_weighted'] + ps['marker_size_scale_weighted'] * weights_norm
            label_corrected = 'Delay Corrected Mag (size by weight)'
            ax.scatter(freqs / 1e9, 20 * np.log10(np.abs(s21_corrected)),
                        s=point_sizes_corrected, color=ps['delay_corrected_color'], alpha=ps['point_alpha'], label=label_corrected)
        else:
            ax.plot(freqs / 1e9, 20 * np.log10(np.abs(s21_corrected)), color=ps['delay_corrected_color'],
                     ls='-', alpha=ps['point_alpha'], lw=ps['line_linewidth'], label=label_corrected)

        if fr_estimate_for_weights is not None and not np.isnan(fr_estimate_for_weights):
            ax.axvline(x=fr_estimate_for_weights / 1e9, color=ps['vline_color'], ls='--',
                        alpha=0.7, label=f'Est. Resonance (for weights)\n{fr_estimate_for_weights/1e9:.4f} GHz')
        
        ax.set_xlabel('Frequency (GHz)', fontsize=ps['axis_label_fontsize'])
        ax.set_ylabel('|S21| (dB)', fontsize=ps['axis_label_fontsize'])
        ax.set_title(f'S21 Amplitude - Delay Correction{title_suffix}', fontsize=ps['title_fontsize'])
        ax.legend(fontsize=ps['legend_fontsize']); ax.grid(True, alpha=ps['grid_alpha'])
        plt.tight_layout();
        return fig

    def plot_normalization_details(self, freqs, s21_before_norm, s21_normalized, norm_params):
        """
        Comprehensive plot for amplitude/phase normalization step.
        """
        ps = self.plot_styles
        figs = [] 

        # Figure 1: Complex plane before and after normalization
        fig1, axes1 = plt.subplots(1, 2, figsize=self.figsize_1x2)
        axes1[0].plot(s21_before_norm.real, s21_before_norm.imag, marker=ps['marker_style'], ls='None',
                     color=ps['delay_corrected_color'], alpha=ps['point_alpha'], 
                     markersize=np.sqrt(ps['marker_size_base']), label='Before Norm')
        if all(k in norm_params for k in ['xc_orig', 'yc_orig', 'radius_orig']) and \
           not any(np.isnan(norm_params[k]) for k in ['xc_orig', 'yc_orig', 'radius_orig'] if isinstance(norm_params[k], float)):
            xc_o, yc_o, r_o = norm_params['xc_orig'], norm_params['yc_orig'], norm_params['radius_orig']
            if r_o > 0:
                theta_c = np.linspace(0, 2 * np.pi, 200)
                circ_o_x, circ_o_y = xc_o + r_o * np.cos(theta_c), yc_o + r_o * np.sin(theta_c)
                axes1[0].plot(circ_o_x, circ_o_y, color=ps['model_fit_color'], lw=ps['fit_linewidth'], label='Fitted Circle')
                axes1[0].plot(xc_o, yc_o, marker='x', color=ps['accent_color_2'], ms=np.sqrt(ps['marker_size_base'])*1.5, label='Center')
        if 'P_off_resonant_orig_coords' in norm_params and not np.any(np.isnan(norm_params['P_off_resonant_orig_coords'])):
            P_off = norm_params['P_off_resonant_orig_coords']
            axes1[0].plot(P_off.real, P_off.imag, marker='*', color=ps['accent_color_1'], 
                          ms=np.sqrt(ps['marker_size_base'])*2, label='P_off (Original Coords)')
        axes1[0].set_title('Before Amp/Phase Correction', fontsize=ps['title_fontsize'])
        axes1[0].set_xlabel('Real'); axes1[0].set_ylabel('Imaginary'); axes1[0].grid(True, alpha=ps['grid_alpha']); axes1[0].axis('equal')
        axes1[0].legend(fontsize=ps['legend_fontsize'])

        axes1[1].plot(s21_normalized.real, s21_normalized.imag, marker=ps['marker_style'], ls='None',
                     color=ps['normalized_data_color'], alpha=ps['point_alpha'], 
                     markersize=np.sqrt(ps['marker_size_base']), label='Normalized Data')
        if 'fit_circle_algebraic_func' in ps:
            xc_n, yc_n, r_n, _ = ps['fit_circle_algebraic_func'](s21_normalized) 
            if not any(np.isnan(v) for v in [xc_n, yc_n, r_n]) and r_n > 0:
                theta_c = np.linspace(0, 2 * np.pi, 200)
                circ_n_x, circ_n_y = xc_n + r_n * np.cos(theta_c), yc_n + r_n * np.sin(theta_c)
                axes1[1].plot(circ_n_x, circ_n_y, color=ps['model_fit_color'], lw=ps['fit_linewidth'], label='Fitted Circle (Norm.)')
                axes1[1].plot(xc_n, yc_n, marker='x', color=ps['accent_color_2'], ms=np.sqrt(ps['marker_size_base'])*1.5, label='Center (Norm.)')
        axes1[1].plot(1, 0, marker='*', color=ps['accent_color_1'], ms=np.sqrt(ps['marker_size_base'])*2, label='Ideal Off Res. (1,0)')
        axes1[1].set_title('After Amp/Phase Correction', fontsize=ps['title_fontsize'])
        axes1[1].set_xlabel('Real'); axes1[1].set_ylabel('Imaginary'); axes1[1].grid(True, alpha=ps['grid_alpha']); axes1[1].axis('equal')
        axes1[1].legend(fontsize=ps['legend_fontsize'])
        plt.tight_layout(); figs.append(fig1)

        fig2, axes2 = plt.subplots(2, 1, figsize=self.figsize_2x1_shared, sharex=True)
        fr_phase = norm_params.get('fr_phase_fit')
        axes2[0].plot(freqs / 1e9, 20 * np.log10(np.abs(s21_before_norm)), color=ps['delay_corrected_color'],
                       ls='-', alpha=ps['point_alpha'], lw=ps['line_linewidth'], label='Before Norm')
        axes2[0].plot(freqs / 1e9, 20 * np.log10(np.abs(s21_normalized)), color=ps['normalized_data_color'],
                       ls='-', alpha=ps['point_alpha'], lw=ps['line_linewidth'], label='Normalized')
        if fr_phase is not None and not np.isnan(fr_phase):
            axes2[0].axvline(x=fr_phase / 1e9, color=ps['vline_color'], ls='--', alpha=0.7, label=f'fr (Phase Fit)\n{fr_phase/1e9:.4f} GHz')
        axes2[0].set_ylabel('|S21| (dB)'); axes2[0].set_title('S21 Amplitude', fontsize=ps['title_fontsize'])
        axes2[0].legend(fontsize=ps['legend_fontsize']); axes2[0].grid(True, alpha=ps['grid_alpha'])
        
        axes2[1].plot(freqs / 1e9, np.unwrap(np.angle(s21_before_norm)), color=ps['delay_corrected_color'],
                       ls='-', alpha=ps['point_alpha'], lw=ps['line_linewidth'], label='Before Norm (Unwrapped)')
        axes2[1].plot(freqs / 1e9, np.unwrap(np.angle(s21_normalized)), color=ps['normalized_data_color'],
                       ls='-', alpha=ps['point_alpha'], lw=ps['line_linewidth'], label='Normalized (Unwrapped)')
        if fr_phase is not None and not np.isnan(fr_phase):
            axes2[1].axvline(x=fr_phase / 1e9, color=ps['vline_color'], ls='--', alpha=0.7)
        axes2[1].set_xlabel('Frequency (GHz)'); axes2[1].set_ylabel('Phase (rad)'); axes2[1].set_title('S21 Phase', fontsize=ps['title_fontsize'])
        axes2[1].legend(fontsize=ps['legend_fontsize']); axes2[1].grid(True, alpha=ps['grid_alpha'])
        plt.tight_layout(); figs.append(fig2)

        if all(k in norm_params for k in ['fr_phase_fit', 'Ql_phase_fit', 'theta0_phase_fit', 
                                           'phase_fit_input_freqs', 'phase_fit_input_centered_phase_data']):
            phase_freqs = norm_params['phase_fit_input_freqs']
            phase_data_centered = norm_params['phase_fit_input_centered_phase_data']
            fr_p, Ql_p, th0_p = norm_params['fr_phase_fit'], norm_params['Ql_phase_fit'], norm_params['theta0_phase_fit']

            if not any(np.isnan(v) for v in [fr_p, Ql_p, th0_p] if isinstance(v, float)):
                phase_model_fitted = phase_model(phase_freqs, fr_p, Ql_p, th0_p)
                residuals_phase = np.unwrap(phase_data_centered) - phase_model_fitted

                fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True) # Keep original figsize
                axes3[0].plot(phase_freqs / 1e9, np.unwrap(phase_data_centered), marker='.', ls='None', color=ps['delay_corrected_color'],
                               alpha=0.6, label='Centered Phase Data (Unwrapped)')
                axes3[0].plot(phase_freqs / 1e9, phase_model_fitted, color=ps['phase_model_color'],
                               lw=ps['model_linewidth'], label='Phase Model Fit')
                axes3[0].axvline(x=fr_p / 1e9, color=ps['vline_color'], ls='--', alpha=0.7, label=f'fr (Phase Fit)\n{fr_p/1e9:.4f} GHz')
                axes3[0].set_ylabel('Phase (rad)'); axes3[0].set_title('Phase Data (Centered) vs. Phase Model Fit', fontsize=ps['title_fontsize'])
                axes3[0].legend(fontsize=ps['legend_fontsize']); axes3[0].grid(True, alpha=ps['grid_alpha'])
                
                axes3[1].plot(phase_freqs / 1e9, residuals_phase, marker='.', ls='None', color=ps['accent_color_1'], alpha=0.6)
                axes3[1].axhline(y=0, color=ps['accent_color_2'], ls='-', alpha=0.5)
                axes3[1].axvline(x=fr_p / 1e9, color=ps['vline_color'], ls='--', alpha=0.7)
                axes3[1].set_xlabel('Frequency (GHz)'); axes3[1].set_ylabel('Residual (rad)'); axes3[1].set_title('Phase Fit Residuals', fontsize=ps['title_fontsize'])
                axes3[1].grid(True, alpha=ps['grid_alpha'])
                plt.tight_layout(); figs.append(fig3)
        
        return figs
    
    def _plot_s21_fitting_panel(self, ax_mag, ax_phase, ax_complex, ax_res, freqs, s21_data, model_data, fr_fit, fit_results, data_color, model_color, data_label_suffix=""):
        """Helper function to create the standard 4-panel S21 fit plot."""
        ps = self.plot_styles
        weights = fit_results.get('weights')

        point_sizes = ps['marker_size_base']
        data_label = f'Data{data_label_suffix}'
        scatter_props = {'marker': ps['marker_style'], 'color': data_color, 'alpha': ps['point_alpha']}

        if weights is not None and len(weights) == len(s21_data) and np.any(weights > 0) and np.max(weights) > 0:
            weights_norm = weights / np.max(weights)
            point_sizes = ps['marker_size_min_weighted'] + ps['marker_size_scale_weighted'] * weights_norm
            data_label = f'Data{data_label_suffix} (size by weight)'
        scatter_props['s'] = point_sizes if isinstance(point_sizes, (int, float)) else np.sqrt(point_sizes)


        # Magnitude
        ax_mag.scatter(freqs / 1e9, 20 * np.log10(np.abs(s21_data)), label=data_label, **scatter_props)
        ax_mag.plot(freqs / 1e9, 20 * np.log10(np.abs(model_data)), 
                      color=model_color, linewidth=ps['model_linewidth'], label='Model')
        if not np.isnan(fr_fit):
            ax_mag.axvline(x=fr_fit / 1e9, color=ps['vline_color'], ls='--', alpha=0.7, label=f'f_r\n{fr_fit/1e9:.4f}GHz')
        ax_mag.set_xlabel('Frequency (GHz)'); ax_mag.set_ylabel('|S21| (dB)')
        ax_mag.grid(True, alpha=ps['grid_alpha']); ax_mag.legend(fontsize=ps['legend_fontsize'])
        
        # Phase
        ax_phase.scatter(freqs / 1e9, np.unwrap(np.angle(s21_data)), label=data_label, **scatter_props)
        ax_phase.plot(freqs / 1e9, np.unwrap(np.angle(model_data)), 
                      color=model_color, linewidth=ps['model_linewidth'], label='Model')
        if not np.isnan(fr_fit):
            ax_phase.axvline(x=fr_fit / 1e9, color=ps['vline_color'], ls='--', alpha=0.7)
        ax_phase.set_xlabel('Frequency (GHz)'); ax_phase.set_ylabel('Phase (rad)')
        ax_phase.grid(True, alpha=ps['grid_alpha']); ax_phase.legend(fontsize=ps['legend_fontsize'])
        
        # Complex Plane
        ax_complex.scatter(s21_data.real, s21_data.imag, label=data_label, **scatter_props)
        ax_complex.plot(model_data.real, model_data.imag, 
                       color=model_color, linewidth=ps['model_linewidth'], label='Model')
        if not np.isnan(fr_fit):
            idx_res = np.argmin(np.abs(freqs - fr_fit))
            if idx_res < len(model_data):
                ax_complex.plot(model_data[idx_res].real, model_data[idx_res].imag, marker='*', 
                              color=ps['accent_color_1'], ms=np.sqrt(ps['marker_size_base'])*2, label='Res. Pt (Model)')
        ax_complex.plot(1, 0, marker='o', color=ps['accent_color_2'], markersize=np.sqrt(ps['marker_size_base']), label='(1,0)')
        ax_complex.set_xlabel('Real'); ax_complex.set_ylabel('Imaginary')
        ax_complex.grid(True, alpha=ps['grid_alpha']); ax_complex.axis('equal')
        ax_complex.legend(fontsize=ps['legend_fontsize'])
        
        # Amplitude Residuals
        residuals_db = 20 * np.log10(np.abs(s21_data)) - 20 * np.log10(np.abs(model_data))
        ax_res.scatter(freqs / 1e9, residuals_db, s=scatter_props.get('s', np.sqrt(ps['marker_size_base'])),
                           marker='.', color=ps['accent_color_1'], alpha=0.6)
        ax_res.axhline(y=0, color=ps['accent_color_2'], ls='-', alpha=0.5)
        if not np.isnan(fr_fit):
            ax_res.axvline(x=fr_fit / 1e9, color=ps['vline_color'], ls='--', alpha=0.7)
        ax_res.set_xlabel('Frequency (GHz)'); ax_res.set_ylabel('Residual (dB)')
        ax_res.set_title('Amplitude Fit Residuals', fontsize=ps['title_fontsize'])
        ax_res.grid(True, alpha=ps['grid_alpha'])

    def plot_fitting_results(self, freqs, s21_data, model_data, fr, title="Fitting Results", weights=None):
        """ Generic S21 fitting results plot. Called if specific plotter is not available."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_2x2)
        fig.suptitle(title, fontsize=self.plot_styles['title_fontsize'] + 2)
        self._plot_s21_fitting_panel(axes[0,0], axes[0,1], axes[1,0], axes[1,1],
                                    freqs, s21_data, model_data, fr, {'weights': weights},
                                    self.plot_styles['normalized_data_color'], 
                                    self.plot_styles['s21_model_color'])
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def plot_dcm_fit_details(self, freqs, s21_data, model_data, fit_results, title="DCM Fitting Results"):
        """Plots specific to DCM fitting results."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_2x2)
        fig.suptitle(title, fontsize=self.plot_styles['title_fontsize'] + 2)
        fr_fit = fit_results.get('fr', np.nan)
        self._plot_s21_fitting_panel(axes[0,0], axes[0,1], axes[1,0], axes[1,1],
                                    freqs, s21_data, model_data, fr_fit, fit_results,
                                    self.plot_styles['normalized_data_color'], 
                                    self.plot_styles['s21_model_color'])
        
        # Add DCM specific annotations to complex plot if desired
        # Example: Ql, Qc_mag, phi from fit_results
        phi_rad = fit_results.get('phi', np.nan)
        Ql = fit_results.get('Ql', np.nan)
        Qc_mag = fit_results.get('Qc_mag', np.nan)
        if not any(np.isnan(v) for v in [phi_rad, Ql, Qc_mag]) and Qc_mag != 0:
            # Calculate circle center and radius from DCM parameters
            radius_dcm = Ql / (2 * Qc_mag)
            xc_dcm = 1 - radius_dcm * np.cos(phi_rad)
            yc_dcm = radius_dcm * np.sin(phi_rad)
            axes[1,0].plot(xc_dcm, yc_dcm, 'x', color='magenta', markersize=8, label=f'DCM Center\n(phi={phi_rad:.2f} rad)')
            # Could draw the DCM circle itself if it's different from the generic fit line
            
        axes[1,0].legend(fontsize=self.plot_styles['legend_fontsize'])
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def plot_cpzm_fit_details(self, freqs, s21_data, model_data, fit_results, title="CPZM Fitting Results"):
        """Plots specific to CPZM fitting results."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_2x2)
        fig.suptitle(title, fontsize=self.plot_styles['title_fontsize'] + 2)
        fr_fit = fit_results.get('fr', np.nan)
        self._plot_s21_fitting_panel(axes[0,0], axes[0,1], axes[1,0], axes[1,1],
                                    freqs, s21_data, model_data, fr_fit, fit_results,
                                    self.plot_styles['normalized_data_color'],
                                    self.plot_styles['s21_model_color'])
        # Add CPZM specific annotations if any. E.g. Qa value if it's informative for the plot.
        # Qa = fit_results.get('Qa', np.nan)
        # axes[1,0].text(0.05, 0.05, f"Qa: {Qa:.1f}", transform=axes[1,0].transAxes, fontsize=self.plot_styles['legend_fontsize'])
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def plot_inverse_fit_details(self, freqs, s21_inv_data, model_inv_data, fit_results, title="Inverse S21 Fitting Results"):
        """Plots specific to Inverse S21 fitting results (plots 1/S21)."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_2x2)
        fig.suptitle(title, fontsize=self.plot_styles['title_fontsize'] + 2)
        ps = self.plot_styles
        fr_fit = fit_results.get('fr', np.nan)
        weights = fit_results.get('weights')

        point_sizes = ps['marker_size_base']
        data_label = '1/S21 Data'
        scatter_props = {'marker': ps['marker_style'], 'color': ps['inverse_data_color'], 'alpha': ps['point_alpha']}

        if weights is not None and len(weights) == len(s21_inv_data) and np.any(weights > 0) and np.max(weights)>0:
            weights_norm = weights / np.max(weights)
            point_sizes = ps['marker_size_min_weighted'] + ps['marker_size_scale_weighted'] * weights_norm
            data_label = '1/S21 Data (size by weight)'
        scatter_props['s'] = point_sizes if isinstance(point_sizes, (int, float)) else np.sqrt(point_sizes)

        # Magnitude of 1/S21
        axes[0,0].scatter(freqs / 1e9, 20 * np.log10(np.abs(s21_inv_data)), label=data_label, **scatter_props)
        axes[0,0].plot(freqs / 1e9, 20 * np.log10(np.abs(model_inv_data)), 
                      color=ps['s21_inv_model_color'], linewidth=ps['model_linewidth'], label='1/S21 Model')
        if not np.isnan(fr_fit):
            axes[0,0].axvline(x=fr_fit / 1e9, color=ps['vline_color'], ls='--', alpha=0.7, label=f'f_r\n{fr_fit/1e9:.4f}GHz')
        axes[0,0].set_xlabel('Frequency (GHz)'); axes[0,0].set_ylabel('|1/S21| (dB)')
        axes[0,0].grid(True, alpha=ps['grid_alpha']); axes[0,0].legend(fontsize=ps['legend_fontsize'])
        
        # Phase of 1/S21
        axes[0,1].scatter(freqs / 1e9, np.unwrap(np.angle(s21_inv_data)), label=data_label, **scatter_props)
        axes[0,1].plot(freqs / 1e9, np.unwrap(np.angle(model_inv_data)), 
                      color=ps['s21_inv_model_color'], linewidth=ps['model_linewidth'], label='1/S21 Model')
        if not np.isnan(fr_fit):
            axes[0,1].axvline(x=fr_fit / 1e9, color=ps['vline_color'], ls='--', alpha=0.7)
        axes[0,1].set_xlabel('Frequency (GHz)'); axes[0,1].set_ylabel('Phase (1/S21) (rad)')
        axes[0,1].grid(True, alpha=ps['grid_alpha']); axes[0,1].legend(fontsize=ps['legend_fontsize'])
        
        # Complex Plane for 1/S21
        axes[1,0].scatter(s21_inv_data.real, s21_inv_data.imag, label=data_label, **scatter_props)
        axes[1,0].plot(model_inv_data.real, model_inv_data.imag, 
                       color=ps['s21_inv_model_color'], linewidth=ps['model_linewidth'], label='1/S21 Model')
        if not np.isnan(fr_fit):
            idx_res = np.argmin(np.abs(freqs - fr_fit))
            if idx_res < len(model_inv_data):
                axes[1,0].plot(model_inv_data[idx_res].real, model_inv_data[idx_res].imag, marker='*', 
                              color=ps['accent_color_1'], ms=np.sqrt(ps['marker_size_base'])*2, label='Res. Pt (Model)')
        axes[1,0].plot(1, 0, marker='o', color=ps['accent_color_2'], markersize=np.sqrt(ps['marker_size_base']), label='Off Res. (1,0)') # S21_inv also approaches 1 off resonance
        axes[1,0].set_xlabel('Real(1/S21)'); axes[1,0].set_ylabel('Imag(1/S21)')
        axes[1,0].grid(True, alpha=ps['grid_alpha']); axes[1,0].axis('equal')
        axes[1,0].legend(fontsize=ps['legend_fontsize'])

        # Amplitude Residuals for 1/S21
        residuals_inv_db = 20 * np.log10(np.abs(s21_inv_data)) - 20 * np.log10(np.abs(model_inv_data))
        axes[1,1].scatter(freqs / 1e9, residuals_inv_db, s=scatter_props.get('s', np.sqrt(ps['marker_size_base'])),
                           marker='.', color=ps['accent_color_1'], alpha=0.6)
        axes[1,1].axhline(y=0, color=ps['accent_color_2'], ls='-', alpha=0.5)
        if not np.isnan(fr_fit):
            axes[1,1].axvline(x=fr_fit / 1e9, color=ps['vline_color'], ls='--', alpha=0.7)
        axes[1,1].set_xlabel('Frequency (GHz)'); axes[1,1].set_ylabel('Residual |1/S21| (dB)')
        axes[1,1].set_title('Inverse S21 Amp. Fit Residuals', fontsize=ps['title_fontsize'])
        axes[1,1].grid(True, alpha=ps['grid_alpha'])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def plot_fitter_weights(self, freqs, weights, fr_est_for_weights, fr_fit, weight_scale_factor, title="Fitting Weights"):
        """Plots the weights used in fitting vs. frequency for any fitter."""
        fig, ax = plt.subplots(figsize=self.figsize_single)
        ps = self.plot_styles

        ax.plot(freqs / 1e9, weights, color=ps['original_data_color'], lw=ps['model_linewidth'])
        if not np.isnan(fr_est_for_weights):
            ax.axvline(x=fr_est_for_weights / 1e9, color=ps['delay_corrected_color'], ls='--',
                         label=f'fr_est (for weights)\n{fr_est_for_weights/1e9:.4f} GHz')
        if not np.isnan(fr_fit):
            ax.axvline(x=fr_fit / 1e9, color=ps['s21_model_color'], ls=':',
                         label=f'fr_fit (final)\n{fr_fit/1e9:.4f} GHz', alpha=0.7)
        
        ax.set_xlabel('Frequency (GHz)', fontsize=ps['axis_label_fontsize'])
        ax.set_ylabel('Weight', fontsize=ps['axis_label_fontsize'])
        title_suffix = f"(scale: {weight_scale_factor:.2f})" if isinstance(weight_scale_factor, (float, int)) else ""
        ax.set_title(f'{title} {title_suffix}', fontsize=ps['title_fontsize'])
        ax.grid(True, alpha=ps['grid_alpha']); ax.legend(fontsize=ps['legend_fontsize'])
        plt.tight_layout()
        return fig

    def add_fit_circle_algebraic_to_plot_styles(self, fit_func):
        """Allows injecting the fit_circle_algebraic function if needed by plot_normalization_details."""
        self.plot_styles['fit_circle_algebraic_func'] = fit_func