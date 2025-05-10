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
                model_s21 = self.fitter.get_model_data(freqs=processed_freqs)
                self._intermediate_results['fitted_model'] = (processed_freqs.copy(), model_s21.copy())

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
        s21_before_step = s21.copy() # To show original vs processed for each step

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
                # plot_normalization_details returns a list of figures
                figs_norm = self.plotter.plot_normalization_details(
                    current_freqs, s21_before_this_preproc, current_s21, norm_params
                )
                for fig_norm in figs_norm:
                    plt.show()
            # Add more elif for other specific preprocessor plots if needed

        # --- 2. Fitting Step ---
        if self.fitter:
            print(f"\nRunning Fitter: {self.fitter.__class__.__name__}...")
            self.results = self.fitter.fit(current_freqs, current_s21) # current_freqs, current_s21 are now fully preprocessed
            self._intermediate_results['final_for_fitter'] = (current_freqs.copy(), current_s21.copy())

            if hasattr(self.fitter, 'get_model_data') and self.results:
                print("Plotting Fitting Results...")
                model_data = self.fitter.get_model_data(current_freqs) # Use freqs that fitter saw
                self._intermediate_results['fitted_model'] = (current_freqs.copy(), model_data.copy())

                fit_title = f"{self.fitter.__class__.__name__} Fit: "
                if 'Qi' in self.results and 'fr' in self.results:
                    fit_title += f"Qi={self.results['Qi']:.0f}, fr={self.results['fr']/1e9:.4f} GHz"
                
                self.plotter.plot_fitting_results(
                    current_freqs, current_s21, model_data,
                    self.results.get('fr', np.nan),
                    title=fit_title,
                    weights=self.results.get('weights') # DCMFitter stores weights in its results
                )
                plt.show()

                # Specific plot for DCMFitter weights
                if self.fitter.__class__.__name__ == "DCMFitter" and \
                   self.results.get('weighted_fit') and self.results.get('weights') is not None:
                    self.plotter.plot_dcm_weights(
                        current_freqs,
                        self.results['weights'],
                        fr_est_for_weights=self.results.get('fr_estimate', np.nan),
                        fr_dcm_fit=self.results.get('fr', np.nan),
                        weight_scale_factor=self.fitter.weight_bandwidth_scale # Assuming fitter has this attribute
                    )
                    plt.show()
            else:
                print("Fitter ran, but model data or results for plotting are unavailable.")
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
            Dictionary containing the (freqs, s21) data of each step, plus 'fitted_model' if available.
        """
        return self._intermediate_results
    
    def get_intermediate_data(self, step_identifier):
        """
        Get the S21 data after a specific preprocessing step or the fitted model.
        
        Parameters
        ----------
        step_identifier : int or str
            If int: Index of the preprocessing step (0 for the first step, etc.)
            If str: Name of the intermediate result key (e.g., 'original', 'CableDelayCorrector_0', 'fitted_model').
        
        Returns
        -------
        tuple
            (freqs, s21_data) after the specified step or for the model.
        
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