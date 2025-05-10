"""
Plotting utilities for resonator data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


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
            Default figure size for plots, by default (12, 9)
        style : str, optional
            Matplotlib style to use, by default 'default'
        """
        self.figsize = figsize
        
        # Set style if provided
        if style != 'default':
            plt.style.use(style)
            
        # Default plot style parameters
        self.plot_styles = {
            'original_data_color': 'blue',
            'delay_corrected_color': 'red',
            'normalized_data_color': 'purple',
            'model_fit_color': 'black',
            'dcm_model_color': 'green',
            'phase_model_color': 'orangered',
            'point_alpha': 0.5,
            'line_linewidth': 1.0,
            'fit_linewidth': 1.5,
            'model_linewidth': 2.0,
            'marker_size_base': 10,
            'marker_size_min_weighted': 5,
            'marker_size_scale_weighted': 30,
            'marker_style': 'o',
            'scatter_cmap_weights': 'viridis',
            'scatter_cmap_residuals': 'coolwarm',
            'axis_label_fontsize': 10,
            'title_fontsize': 12,
            'legend_fontsize': 8,
            'grid_alpha': 0.3,
            'accent_color_1': 'darkgreen',
            'accent_color_2': 'black',
            'vline_color': 'dimgray',
        }
    
    def plot_raw_data(self, freqs, s21, title="Raw S21 Data"):
        """
        Plot the raw S21 data in magnitude/phase format and complex plane.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data in Hz
        s21 : array_like
            Complex S21 data
        title : str, optional
            Plot title, by default "Raw S21 Data"
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plots
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=self.plot_styles['title_fontsize'] + 2)
        
        # Plot amplitude in dB
        axes[0, 0].plot(freqs / 1e9, 20 * np.log10(np.abs(s21)), 
                      color=self.plot_styles['original_data_color'],
                      linewidth=self.plot_styles['line_linewidth'])
        axes[0, 0].set_xlabel('Frequency (GHz)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[0, 0].set_ylabel('|S21| (dB)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[0, 0].set_title('Magnitude', fontsize=self.plot_styles['title_fontsize'])
        axes[0, 0].grid(True, alpha=self.plot_styles['grid_alpha'])
        
        # Plot phase in radians
        axes[0, 1].plot(freqs / 1e9, np.unwrap(np.angle(s21)),
                      color=self.plot_styles['original_data_color'],
                      linewidth=self.plot_styles['line_linewidth'])
        axes[0, 1].set_xlabel('Frequency (GHz)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[0, 1].set_ylabel('Phase (rad)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[0, 1].set_title('Unwrapped Phase', fontsize=self.plot_styles['title_fontsize'])
        axes[0, 1].grid(True, alpha=self.plot_styles['grid_alpha'])
        
        # Plot in complex plane
        axes[1, 0].plot(s21.real, s21.imag, 
                       marker=self.plot_styles['marker_style'], 
                       ls='None',
                       color=self.plot_styles['original_data_color'], 
                       alpha=self.plot_styles['point_alpha'])
        axes[1, 0].set_xlabel('Real', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[1, 0].set_ylabel('Imaginary', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[1, 0].set_title('Complex Plane', fontsize=self.plot_styles['title_fontsize'])
        axes[1, 0].grid(True, alpha=self.plot_styles['grid_alpha'])
        axes[1, 0].axis('equal')
        
        # Plot phase vs frequency
        axes[1, 1].plot(freqs / 1e9, np.angle(s21),
                      color=self.plot_styles['original_data_color'],
                      linewidth=self.plot_styles['line_linewidth'])
        axes[1, 1].set_xlabel('Frequency (GHz)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[1, 1].set_ylabel('Phase (rad)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[1, 1].set_title('Phase', fontsize=self.plot_styles['title_fontsize'])
        axes[1, 1].grid(True, alpha=self.plot_styles['grid_alpha'])
        
        plt.tight_layout()
        return fig
    
    def plot_delay_correction(self, freqs, s21_original, s21_corrected, delay_ns, fitted_circle=None):
        """
        Plot the original and delay-corrected S21 data.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data in Hz
        s21_original : array_like
            Original complex S21 data
        s21_corrected : array_like
            Delay-corrected complex S21 data
        delay_ns : float
            Cable delay in nanoseconds
        fitted_circle : tuple, optional
            Tuple (xc, yc, r) containing the circle center and radius, by default None
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plots
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot original data
        axes[0].plot(s21_original.real, s21_original.imag, 
                   marker=self.plot_styles['marker_style'], 
                   ls='None',
                   color=self.plot_styles['original_data_color'], 
                   alpha=self.plot_styles['point_alpha'], 
                   label='Original Data')
        axes[0].set_title('Before Delay Correction', fontsize=self.plot_styles['title_fontsize'])
        axes[0].set_xlabel('Real', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[0].set_ylabel('Imaginary', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[0].grid(True, alpha=self.plot_styles['grid_alpha'])
        axes[0].axis('equal')
        axes[0].legend(fontsize=self.plot_styles['legend_fontsize'])
        
        # Plot corrected data
        axes[1].plot(s21_corrected.real, s21_corrected.imag, 
                   marker=self.plot_styles['marker_style'], 
                   ls='None',
                   color=self.plot_styles['delay_corrected_color'], 
                   alpha=self.plot_styles['point_alpha'], 
                   label='Delay Corrected Data')
        
        # Plot fitted circle if provided
        if fitted_circle is not None:
            xc, yc, r = fitted_circle
            theta_circle = np.linspace(0, 2 * np.pi, 200)
            circle_x = xc + r * np.cos(theta_circle)
            circle_y = yc + r * np.sin(theta_circle)
            axes[1].plot(circle_x, circle_y, 
                       color=self.plot_styles['model_fit_color'],
                       lw=self.plot_styles['fit_linewidth'], 
                       label='Fitted Circle')
            axes[1].plot(xc, yc, marker='x', 
                       color=self.plot_styles['accent_color_2'],
                       ms=self.plot_styles['marker_size_base'], 
                       label='Circle Center')
            
        axes[1].plot(0, 0, marker='+', 
                   color=self.plot_styles['accent_color_2'], 
                   ms=self.plot_styles['marker_size_base'], 
                   label='Origin')
        axes[1].set_title(f'After Delay Correction (delay = {delay_ns:.3f} ns)', 
                        fontsize=self.plot_styles['title_fontsize'])
        axes[1].set_xlabel('Real', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[1].set_ylabel('Imaginary', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[1].grid(True, alpha=self.plot_styles['grid_alpha'])
        axes[1].axis('equal')
        axes[1].legend(fontsize=self.plot_styles['legend_fontsize'])
        
        plt.tight_layout()
        return fig
    
    def plot_fitting_results(self, freqs, s21_data, model_data, fr, title="Fitting Results"):
        """
        Plot the fitting results compared to the data.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data in Hz
        s21_data : array_like
            Measured complex S21 data
        model_data : array_like
            Fitted model complex S21 data
        fr : float
            Resonance frequency in Hz
        title : str, optional
            Plot title, by default "Fitting Results"
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plots
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=self.plot_styles['title_fontsize'] + 2)
        
        # Plot magnitude
        axes[0, 0].plot(freqs / 1e9, 20 * np.log10(np.abs(s21_data)), 
                      marker=self.plot_styles['marker_style'], ls='None',
                      color=self.plot_styles['normalized_data_color'], 
                      alpha=self.plot_styles['point_alpha'], label='Data')
        axes[0, 0].plot(freqs / 1e9, 20 * np.log10(np.abs(model_data)), 
                      color=self.plot_styles['dcm_model_color'],
                      linewidth=self.plot_styles['model_linewidth'], label='Model')
        axes[0, 0].axvline(x=fr / 1e9, color=self.plot_styles['vline_color'], 
                         ls='--', alpha=0.7, label='Resonance')
        axes[0, 0].set_xlabel('Frequency (GHz)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[0, 0].set_ylabel('|S21| (dB)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[0, 0].grid(True, alpha=self.plot_styles['grid_alpha'])
        axes[0, 0].legend(fontsize=self.plot_styles['legend_fontsize'])
        
        # Plot phase
        axes[0, 1].plot(freqs / 1e9, np.unwrap(np.angle(s21_data)), 
                      marker=self.plot_styles['marker_style'], ls='None',
                      color=self.plot_styles['normalized_data_color'], 
                      alpha=self.plot_styles['point_alpha'], label='Data')
        axes[0, 1].plot(freqs / 1e9, np.unwrap(np.angle(model_data)), 
                      color=self.plot_styles['dcm_model_color'],
                      linewidth=self.plot_styles['model_linewidth'], label='Model')
        axes[0, 1].axvline(x=fr / 1e9, color=self.plot_styles['vline_color'], 
                         ls='--', alpha=0.7, label='Resonance')
        axes[0, 1].set_xlabel('Frequency (GHz)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[0, 1].set_ylabel('Phase (rad)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[0, 1].grid(True, alpha=self.plot_styles['grid_alpha'])
        axes[0, 1].legend(fontsize=self.plot_styles['legend_fontsize'])
        
        # Plot in complex plane
        axes[1, 0].plot(s21_data.real, s21_data.imag, 
                       marker=self.plot_styles['marker_style'], ls='None',
                       color=self.plot_styles['normalized_data_color'], 
                       alpha=self.plot_styles['point_alpha'], label='Data')
        axes[1, 0].plot(model_data.real, model_data.imag, 
                       color=self.plot_styles['dcm_model_color'],
                       linewidth=self.plot_styles['model_linewidth'], label='Model')
        
        # Mark resonance point
        idx_res = np.argmin(np.abs(freqs - fr))
        if idx_res < len(model_data):
            axes[1, 0].plot(model_data[idx_res].real, model_data[idx_res].imag, 
                          marker='*', color=self.plot_styles['accent_color_1'],
                          ms=self.plot_styles['marker_size_base']+2, label='Resonance')
        
        # Mark off-resonance point
        axes[1, 0].plot(1, 0, marker='o', color=self.plot_styles['accent_color_2'],
                       ms=self.plot_styles['marker_size_base']-2, label='Off Res. (1,0)')
        
        axes[1, 0].set_xlabel('Real', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[1, 0].set_ylabel('Imaginary', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[1, 0].grid(True, alpha=self.plot_styles['grid_alpha'])
        axes[1, 0].axis('equal')
        axes[1, 0].legend(fontsize=self.plot_styles['legend_fontsize'])
        
        # Plot residuals
        residuals_db = 20 * np.log10(np.abs(s21_data)) - 20 * np.log10(np.abs(model_data))
        axes[1, 1].plot(freqs / 1e9, residuals_db,
                      marker='.', ls='None', color=self.plot_styles['accent_color_1'], 
                      alpha=0.6)
        axes[1, 1].axhline(y=0, color=self.plot_styles['accent_color_2'], ls='-', alpha=0.5)
        axes[1, 1].axvline(x=fr / 1e9, color=self.plot_styles['vline_color'], 
                         ls='--', alpha=0.7)
        axes[1, 1].set_xlabel('Frequency (GHz)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[1, 1].set_ylabel('Residual (dB)', fontsize=self.plot_styles['axis_label_fontsize'])
        axes[1, 1].set_title('Amplitude Fit Residuals', fontsize=self.plot_styles['title_fontsize'])
        axes[1, 1].grid(True, alpha=self.plot_styles['grid_alpha'])
        
        plt.tight_layout()
        return fig
