"""
Base implementations for preprocessing modules.
"""

import numpy as np
from resonfit.core.base import BasePreprocessor


class FunctionPreprocessor(BasePreprocessor):
    """
    A simple preprocessor that applies a user-defined function to S21 data.
    
    This class allows users to quickly create custom preprocessors by
    providing a function that modifies S21 data.
    
    Attributes
    ----------
    func : callable
        Function to apply to S21 data
    name : str
        Name of the preprocessor
    """
    
    def __init__(self, func, name="CustomPreprocessor"):
        """
        Initialize with a processing function.
        
        Parameters
        ----------
        func : callable
            Function that takes (freqs, s21) and returns processed (freqs, s21)
        name : str, optional
            Name of the preprocessor. Default is "CustomPreprocessor".
        """
        self._func = func
        self._name = name
    
    def preprocess(self, freqs, s21):
        """
        Apply the function to S21 data.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data
        s21 : array_like
            Complex S21 data
            
        Returns
        -------
        tuple
            Processed (freqs, s21) tuple
        """
        return self._func(freqs, s21)
    
    @property
    def name(self):
        """Return the name of the preprocessor."""
        return self._name
