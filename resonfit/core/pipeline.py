"""
Pipeline for resonator data processing and fitting.

This module provides the ResonatorPipeline class which orchestrates the
preprocessing and fitting steps for resonator data analysis.
"""

import numpy as np


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
        self._intermediate_results = {}
        self._original_data = None
    
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
    
    def run(self, freqs, s21, plot=False):
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
            Whether to generate plots during the run, by default False
        
        Returns
        -------
        dict
            Results from the fitting process
        
        Raises
        ------
        ValueError
            If no fitter has been set
        """
        if not self.preprocessors and not self.fitter:
            raise ValueError("Pipeline is empty. Add preprocessors and a fitter before running.")
        
        # Save original data
        self._original_data = (np.array(freqs), np.array(s21))
        
        # Apply preprocessing steps
        processed_freqs, processed_s21 = np.array(freqs).copy(), np.array(s21).copy()
        self._intermediate_results = {'original': (processed_freqs.copy(), processed_s21.copy())}
        
        for i, preprocessor in enumerate(self.preprocessors):
            processed_freqs, processed_s21 = preprocessor.preprocess(processed_freqs, processed_s21)
            step_name = f"{preprocessor.__class__.__name__}_{i}"
            self._intermediate_results[step_name] = (processed_freqs.copy(), processed_s21.copy())
        
        # Execute fitting if a fitter is available
        if self.fitter:
            self.results = self.fitter.fit(processed_freqs, processed_s21)
            self._intermediate_results['final'] = (processed_freqs, processed_s21)
        else:
            self.results = {"warning": "No fitter set, returning preprocessed data only"}
        
        # Generate plots if requested
        if plot:
            self._generate_plots()
        
        # Return the results
        return self.results
    
    def get_intermediate_results(self):
        """
        Get the intermediate results from the pipeline run.
        
        Returns
        -------
        dict
            Dictionary containing the results of each step
        """
        return self._intermediate_results
    
    def get_intermediate_data(self, index):
        """
        Get the S21 data after a specific preprocessing step.
        
        Parameters
        ----------
        index : int
            Index of the preprocessing step (0 for the first step, etc.)
        
        Returns
        -------
        array_like
            S21 data after the specified preprocessing step
        
        Raises
        ------
        IndexError
            If index is out of range
        """
        if not self._intermediate_results:
            raise ValueError("No intermediate results available. Run the pipeline first.")
        
        if index < 0 or index >= len(self.preprocessors):
            raise IndexError(f"Index {index} out of range. Pipeline has {len(self.preprocessors)} preprocessing steps.")
        
        # Get the specified preprocessor's output
        preprocessor = self.preprocessors[index]
        step_name = f"{preprocessor.__class__.__name__}_{index}"
        
        if step_name in self._intermediate_results:
            # Return only the s21 data (second element of the tuple)
            return self._intermediate_results[step_name][1]
        
        # Fallback: return the final preprocessed data
        for i in range(index, -1, -1):
            step_name = f"{self.preprocessors[i].__class__.__name__}_{i}"
            if step_name in self._intermediate_results:
                return self._intermediate_results[step_name][1]
        
        # If no intermediate result is found, return the original data
        return self._intermediate_results.get('original', (None, None))[1]
    
    def _generate_plots(self):
        """Generate plots of the pipeline results."""
        # This will be implemented when the visualization module is added
        pass