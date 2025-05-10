"""
Pipeline for orchestrating resonator data analysis.

This module provides the ResonatorPipeline class, which allows users to
combine different preprocessing and fitting methods into a custom workflow.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from resonfit.core.base import BasePreprocessor, BaseFitter


class ResonatorPipeline:
    """
    Pipeline for combining preprocessing and fitting steps.
    
    This class allows users to build a custom analysis pipeline by
    adding preprocessing steps and a fitting method, then execute
    the entire workflow with a single call.
    
    Attributes
    ----------
    preprocessors : list
        List of preprocessor objects
    fitter : BaseFitter or None
        Fitter object
    results : dict
        Results from the last pipeline run
    """
    
    def __init__(self):
        """Initialize the pipeline with empty components."""
        self.preprocessors = []
        self.fitter = None
        self.results = {}
        self._original_data = None
        self._processed_data = None
    
    def add_preprocessor(self, preprocessor: BasePreprocessor):
        """
        Add a preprocessing step to the pipeline.
        
        Parameters
        ----------
        preprocessor : BasePreprocessor
            Preprocessor object to add
            
        Returns
        -------
        ResonatorPipeline
            Self, for method chaining
        """
        if not isinstance(preprocessor, BasePreprocessor):
            raise TypeError("Preprocessor must be an instance of BasePreprocessor")
        
        self.preprocessors.append(preprocessor)
        return self
    
    def set_fitter(self, fitter: BaseFitter):
        """
        Set the fitting method for the pipeline.
        
        Parameters
        ----------
        fitter : BaseFitter
            Fitter object to use
            
        Returns
        -------
        ResonatorPipeline
            Self, for method chaining
        """
        if not isinstance(fitter, BaseFitter):
            raise TypeError("Fitter must be an instance of BaseFitter")
        
        self.fitter = fitter
        return self
    
    def run(self, freqs: np.ndarray, s21: np.ndarray, plot: bool = False) -> Dict:
        """
        Run the pipeline on the provided data.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data array (Hz)
        s21 : array_like
            Complex S21 transmission data
        plot : bool, optional
            Whether to generate plots during the analysis. Default is False.
            
        Returns
        -------
        dict
            Dictionary containing fitting results
            
        Raises
        ------
        ValueError
            If no fitter has been set
        """
        # Store original data
        self._original_data = (np.asarray(freqs), np.asarray(s21))
        
        # Execute preprocessing steps
        processed_freqs, processed_s21 = freqs.copy(), s21.copy()
        preprocessing_results = []
        
        for preprocessor in self.preprocessors:
            try:
                processed_freqs, processed_s21 = preprocessor.preprocess(
                    processed_freqs, processed_s21
                )
                preprocessing_results.append({
                    "name": preprocessor.name,
                    "parameters": preprocessor.parameters
                })
            except Exception as e:
                raise RuntimeError(
                    f"Error in preprocessor '{preprocessor.name}': {str(e)}"
                ) from e
        
        self._processed_data = (processed_freqs, processed_s21)
        
        # Execute fitting step
        if not self.fitter:
            raise ValueError("No fitter has been set. Use set_fitter() to set a fitter.")
        
        try:
            fitting_results = self.fitter.fit(processed_freqs, processed_s21)
        except Exception as e:
            raise RuntimeError(
                f"Error in fitter '{self.fitter.name}': {str(e)}"
            ) from e
        
        # Store all results
        self.results = {
            "preprocessing": preprocessing_results,
            "fitting": {
                "name": self.fitter.name,
                "parameters": self.fitter.parameters,
                **fitting_results
            }
        }
        
        return self.results
    
    def get_model_data(self, freqs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get model data from the fitter.
        
        Parameters
        ----------
        freqs : array_like, optional
            Frequency data array (Hz). If None, uses frequencies from the last run.
            
        Returns
        -------
        array_like
            Complex model S21 data
            
        Raises
        ------
        ValueError
            If no fitter has been set or no data has been processed
        """
        if not self.fitter:
            raise ValueError("No fitter has been set. Use set_fitter() to set a fitter.")
        
        if freqs is None:
            if self._processed_data is None:
                raise ValueError("No data has been processed. Run the pipeline first.")
            freqs = self._processed_data[0]
        
        return self.fitter.get_model_data(freqs)
    
    def get_original_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the original data from the last run.
        
        Returns
        -------
        tuple
            (freqs, s21) from the last run
            
        Raises
        ------
        ValueError
            If no data has been processed
        """
        if self._original_data is None:
            raise ValueError("No data has been processed. Run the pipeline first.")
        
        return self._original_data
    
    def get_processed_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the processed data from the last run.
        
        Returns
        -------
        tuple
            (freqs, s21) after all preprocessing steps
            
        Raises
        ------
        ValueError
            If no data has been processed
        """
        if self._processed_data is None:
            raise ValueError("No data has been processed. Run the pipeline first.")
        
        return self._processed_data
    
    def clear(self):
        """
        Clear all preprocessors, fitter, and results.
        
        Returns
        -------
        ResonatorPipeline
            Self, for method chaining
        """
        self.preprocessors = []
        self.fitter = None
        self.results = {}
        self._original_data = None
        self._processed_data = None
        return self
