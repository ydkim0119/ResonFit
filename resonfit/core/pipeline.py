"""
Pipeline for resonator data processing and fitting.

This module provides the ResonatorPipeline class which orchestrates the
preprocessing and fitting steps for resonator data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt # For showing plots
from ..visualization import ResonancePlotter
from ..preprocessing.base import fit_circle_algebraic # For plotter if needed


class ResonatorPipeline:
    """
    Pipeline for combining preprocessing and fitting steps.
    
    This class allows users to combine multiple preprocessing steps
    with a fitting method to create a customized analysis workflow.
    
    Attributes
    ----------
    preprocessors : list
        List of preprocessor objects that implement the BasePreprocessor interface
    fitter : object
        Fitter object that implements the BaseFitter interface
    results : dict
        Dictionary containing the results of the last run
    """
    
    def __init__(self):
        """Initialize an empty pipeline."""
        self.preprocessors = []
        self.fitter = None
        self.results = {}
        self._intermediate_results = {} # Stores (freqs, s21) tuples after each step
        self._original_data = None
        self.plotter = None # Will be initialized when needed
    
    def add_preprocessor(self, preprocessor):
        """
        Add a preprocessing step to the pipeline.
        
        Parameters
        ----------
        preprocessor : BasePreprocessor
            A preprocessor object that implements the BasePreprocessor interface
        
        Returns
        -------
        self : ResonatorPipeline
            Returns self for method chaining
        """
        self.preprocessors.append(preprocessor)
        return self
    
    def set_fitter(self, fitter):
        """
        Set the fitting method for the pipeline.
        
        Parameters
        ----------
        fitter : BaseFitter
            A fitter object that implements the BaseFitter interface
        
        Returns
        -------
        self : ResonatorPipeline
            Returns self for method chaining
        """
        self.fitter = fitter
        return self
    
    def run(self, freqs, s21, plot=False): # Kept for backward compatibility or simple runs
        """
        Run the pipeline on the given data.
        
        This method applies each preprocessing step in sequence, then
        runs the fitting method on the processed data.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
        s21 : array_like
            Complex S21 data
        plot : bool, optional
            If True, calls run_analysis_and_plot. Kept for simplicity, but
            run_analysis_and_plot offers more direct control. Default is False.
        
        Returns
        -------
        dict
            Results from the fitting process
        
        Raises
        ------
        ValueError
            If no fitter has been set and preprocessors are also empty.
        """
        if not self.preprocessors and not self.fitter:
            raise ValueError("Pipeline is empty. Add preprocessors and/or a fitter before running.")

        if plot: # If plot is True, delegate to the more comprehensive plotting method
            print("Plotting enabled, calling run_analysis_and_plot().")
            return self.run_analysis_and_plot(freqs, s21)

        # Save original data
        self._original_data = (np.array(freqs), np.array(s21))
        
        processed_freqs, processed_s21 = np.array(freqs).copy(), np.array(s21).copy()
        self._intermediate_results['original'] = (processed_freqs.copy(), processed_s21.copy())
        
        for i, preprocessor in enumerate(self.preprocessors):
            processed_freqs, processed_s21 = preprocessor.preprocess(processed_freqs, processed_s21)
            step_name = f"{preprocessor.__class__.__name__}_{i}"
            self._intermediate_results[step_name] = (processed_freqs.copy(), processed_s21.copy())
        
        if self.fitter:
            self.results = self.fitter.fit(processed_freqs, processed_s21)
            self._intermediate_results['final_for_fitter'] = (processed_freqs.copy(), processed_s21.copy())
            if hasattr(self.fitter, 'get_model_data') and hasattr(self.fitter, 'fit_results') and self.fitter.fit_results:
                 # Store model data if available, using the freqs that were fed to the fitter
                model_s21_fitted = self.fitter.get_model_data(freqs=processed_freqs)
                self._intermediate_results['fitted_model_s21'] = (processed_freqs.copy(), model_s21_fitted.copy())
                if hasattr(self.fitter, 'get_inverse_model_data'): # Specific for InverseFitter
                    model_s21_inv_fitted = self.fitter.get_inverse_model_data(freqs=processed_freqs)
                    self._intermediate_results['fitted_model_s21_inv'] = (processed_freqs.copy(), model_s21_inv_fitted.copy())


        else:
            self.results = {"warning": "No fitter set. Preprocessing complete."}
            self._intermediate_results['final_preprocessed_only'] = (processed_freqs.copy(), processed_s21.copy())
        
        return self.results

    def run_analysis_and_plot(self, freqs, s21):
        """
        Run the full analysis pipeline and generate plots for each significant step.

        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
        s21 : array_like
            Complex S21 data

        Returns
        -------
        dict
            Results from the fitting process.
        """
        if self.plotter is None:
            self.plotter = ResonancePlotter()
            # Inject fit_circle_algebraic if plotter needs it for normalization plots
            if hasattr(self.plotter, 'add_fit_circle_algebraic_to_plot_styles'):
                 self.plotter.add_fit_circle_algebraic_to_plot_styles(fit_circle_algebraic)


        print("--- Starting Full Analysis with Plotting ---")
        
        # --- 0. Original Data ---
        print("\nPlotting Raw Data...")
        self._original_data = (np.array(freqs).copy(), np.array(s21).copy())
        self._intermediate_results['original'] = self._original_data
        self.plotter.plot_raw_data(freqs, s21, title="Raw S21 Data")
        plt.show()

        current_freqs, current_s21 = freqs.copy(), s21.copy()
        # s21_before_step = s21.copy() # To show original vs processed for each step. Used locally per preprocessor.

        # --- 1. Preprocessing Steps ---
        for i, preprocessor in enumerate(self.preprocessors):
            print(f"\nRunning Preprocessor: {preprocessor.__class__.__name__}...")
            s21_before_this_preproc = current_s21.copy() # Data before this specific preprocessor
            
            current_freqs, current_s21 = preprocessor.preprocess(current_freqs, current_s21)
            step_name = f"{preprocessor.__class__.__name__}_{i}"
            self._intermediate_results[step_name] = (current_freqs.copy(), current_s21.copy())

            # Plotting for specific preprocessors
            if preprocessor.__class__.__name__ == "CableDelayCorrector":
                print("Plotting Cable Delay Correction Results...")
                delay_params = preprocessor.get_final_params_for_plotting()
                optimal_delay_ns = preprocessor.get_delay() * 1e9
                
                self.plotter.plot_delay_correction(
                    current_freqs, s21_before_this_preproc, current_s21,
                    optimal_delay_ns,
                    corrected_circle_params=delay_params['circle_params'],
                    weights=delay_params['weights']
                )
                plt.show()
                self.plotter.plot_delay_correction_magnitude_vs_freq(
                    current_freqs, s21_before_this_preproc, current_s21,
                    weights=delay_params['weights'],
                    fr_estimate_for_weights=delay_params['fr_estimate_for_weights']
                )
                plt.show()

            elif preprocessor.__class__.__name__ == "AmplitudePhaseNormalizer":
                print("Plotting Amplitude/Phase Normalization Results...")
                norm_params = preprocessor.get_normalization_params()
                figs_norm = self.plotter.plot_normalization_details(
                    current_freqs, s21_before_this_preproc, current_s21, norm_params
                )
                for fig_norm in figs_norm: # plot_normalization_details now returns a list of figures
                    plt.show()

        # --- 2. Fitting Step ---
        if self.fitter:
            print(f"\nRunning Fitter: {self.fitter.__class__.__name__}...")
            # For InverseFitter, it needs original S21, but it will invert it internally.
            # Other fitters expect preprocessed S21 directly.
            s21_for_fitter = current_s21
            if self.fitter.__class__.__name__ == "InverseFitter":
                # InverseFitter takes s21 and inverts it internally.
                # The 'final_for_fitter' should be what the fitter *sees* before its own processing
                self.results = self.fitter.fit(current_freqs, s21_for_fitter)
                self._intermediate_results['final_for_fitter_s21'] = (current_freqs.copy(), s21_for_fitter.copy())
                if 's21_inv_data_for_fit' in self.results:
                     self._intermediate_results['final_for_fitter_s21_inv'] = (current_freqs.copy(), self.results['s21_inv_data_for_fit'].copy())
            else:
                self.results = self.fitter.fit(current_freqs, s21_for_fitter)
                self._intermediate_results['final_for_fitter'] = (current_freqs.copy(), s21_for_fitter.copy())


            if self.results and 'fr' in self.results: # Check if fit was successful enough to produce fr
                print("Plotting Fitting Results...")
                
                # Common parameters for plot titles
                fit_title_base = f"{self.fitter.__class__.__name__} Fit"
                param_summary = []
                if 'fr' in self.results: param_summary.append(f"fr={self.results['fr']/1e9:.4f} GHz")
                if 'Qi' in self.results: param_summary.append(f"Qi={self.results['Qi']:.0f}")
                if 'Ql' in self.results: param_summary.append(f"Ql={self.results['Ql']:.0f}")
                if 'Qc_mag' in self.results: param_summary.append(f"|Qc|={self.results['Qc_mag']:.0f}") # For DCM
                if 'Qc' in self.results and self.fitter.__class__.__name__ == "CPZMFitter": param_summary.append(f"Qc={self.results['Qc']:.0f}") # For CPZM
                if 'Qc_star' in self.results: param_summary.append(f"Qc*={self.results['Qc_star']:.0f}") # For Inverse
                
                fit_title = f"{fit_title_base}: {', '.join(param_summary)}"

                # Fitter-specific plotting
                if self.fitter.__class__.__name__ == "DCMFitter":
                    model_s21_fitted = self.fitter.get_model_data(current_freqs)
                    self._intermediate_results['fitted_model_s21'] = (current_freqs.copy(), model_s21_fitted.copy())
                    self.plotter.plot_dcm_fit_details(
                        current_freqs, s21_for_fitter, model_s21_fitted, self.results, fit_title
                    )
                    plt.show()
                    if self.results.get('weighted_fit') and self.results.get('weights') is not None:
                        self.plotter.plot_fitter_weights( # Generic weight plotter
                            current_freqs, self.results['weights'],
                            self.results.get('fr_estimate', np.nan), self.results.get('fr', np.nan),
                            getattr(self.fitter, 'weight_bandwidth_scale', 'N/A'),
                            title=f"DCM Fitter Weights"
                        )
                        plt.show()

                elif self.fitter.__class__.__name__ == "InverseFitter":
                    model_s21_inv_fitted = self.fitter.get_inverse_model_data(current_freqs)
                    s21_inv_data = self.results.get('s21_inv_data_for_fit')
                    if s21_inv_data is None: # Should have been stored by fitter
                        s21_inv_data = 1.0 / s21_for_fitter
                    self._intermediate_results['fitted_model_s21_inv'] = (current_freqs.copy(), model_s21_inv_fitted.copy())
                    self.plotter.plot_inverse_fit_details(
                        current_freqs, s21_inv_data, model_s21_inv_fitted, self.results, fit_title
                    )
                    plt.show()
                    if self.results.get('weighted_fit') and self.results.get('weights') is not None:
                        self.plotter.plot_fitter_weights(
                            current_freqs, self.results['weights'],
                            self.results.get('fr_estimate', np.nan), self.results.get('fr', np.nan),
                            getattr(self.fitter, 'weight_bandwidth_scale', 'N/A'),
                            title=f"Inverse Fitter Weights (for 1/S21)"
                        )
                        plt.show()
                
                elif self.fitter.__class__.__name__ == "CPZMFitter":
                    model_s21_fitted = self.fitter.get_model_data(current_freqs)
                    self._intermediate_results['fitted_model_s21'] = (current_freqs.copy(), model_s21_fitted.copy())
                    self.plotter.plot_cpzm_fit_details(
                        current_freqs, s21_for_fitter, model_s21_fitted, self.results, fit_title
                    )
                    plt.show()
                    if self.results.get('weighted_fit') and self.results.get('weights') is not None:
                        self.plotter.plot_fitter_weights(
                            current_freqs, self.results['weights'],
                            self.results.get('fr_estimate', np.nan), self.results.get('fr', np.nan),
                            getattr(self.fitter, 'weight_bandwidth_scale', 'N/A'),
                            title=f"CPZM Fitter Weights"
                        )
                        plt.show()
                else: # Fallback to generic S21 plotter
                    if hasattr(self.fitter, 'get_model_data'):
                        model_s21_fitted = self.fitter.get_model_data(current_freqs)
                        self._intermediate_results['fitted_model_s21'] = (current_freqs.copy(), model_s21_fitted.copy())
                        self.plotter.plot_fitting_results( # Generic S21 plot
                            current_freqs, s21_for_fitter, model_s21_fitted,
                            self.results.get('fr', np.nan),
                            title=fit_title,
                            weights=self.results.get('weights')
                        )
                        plt.show()
                    else:
                        print("Fitter ran, but model data for S21 plotting is unavailable.")
            else:
                 print(f"Fitter {self.fitter.__class__.__name__} ran, but results for plotting are incomplete or fit failed.")
        else:
            self.results = {"warning": "No fitter set. Preprocessing complete."}
            self._intermediate_results['final_preprocessed_only'] = (current_freqs.copy(), current_s21.copy())
            print("\nNo fitter set. Analysis complete after preprocessing.")

        print("\n--- Full Analysis with Plotting Complete ---")
        return self.results

    def get_intermediate_results(self):
        """
        Get the intermediate results from the pipeline run.
        
        Returns
        -------
        dict
            Dictionary containing the (freqs, s21) data of each step, plus 'fitted_model_s21' 
            and/or 'fitted_model_s21_inv' if available.
        """
        return self._intermediate_results
    
    def get_intermediate_data(self, step_identifier):
        """
        Get the S21 (or 1/S21 for specific keys) data after a specific step or the fitted model.
        
        Parameters
        ----------
        step_identifier : int or str
            If int: Index of the preprocessing step (0 for the first step, etc.)
            If str: Name of the intermediate result key (e.g., 'original', 
                    'CableDelayCorrector_0', 'fitted_model_s21', 'fitted_model_s21_inv').
        
        Returns
        -------
        tuple
            (freqs, data) after the specified step or for the model.
        
        Raises
        ------
        KeyError or IndexError
            If the identifier is not found or index is out of range.
        ValueError
            If no intermediate results available.
        """
        if not self._intermediate_results:
            raise ValueError("No intermediate results available. Run the pipeline first.")
        
        if isinstance(step_identifier, int):
            if step_identifier < 0 or step_identifier >= len(self.preprocessors):
                raise IndexError(f"Preprocessor index {step_identifier} out of range. Pipeline has {len(self.preprocessors)} preprocessors.")
            preprocessor = self.preprocessors[step_identifier]
            key_to_find = f"{preprocessor.__class__.__name__}_{step_identifier}"
            if key_to_find not in self._intermediate_results: # Should not happen if logic is correct
                raise KeyError(f"Internal key '{key_to_find}' for preprocessor index {step_identifier} not found.")
            return self._intermediate_results[key_to_find]
        elif isinstance(step_identifier, str):
            if step_identifier not in self._intermediate_results:
                raise KeyError(f"Step identifier '{step_identifier}' not found in intermediate results. Available: {list(self._intermediate_results.keys())}")
            return self._intermediate_results[step_identifier]
        else:
            raise TypeError("step_identifier must be an integer index or a string key.")

    def _generate_plots(self): # Legacy, now handled by run_analysis_and_plot
        """Generate plots of the pipeline results."""
        # This was a placeholder. The new run_analysis_and_plot method handles plotting.
        print("Plotting is handled by run_analysis_and_plot() method.")
        pass