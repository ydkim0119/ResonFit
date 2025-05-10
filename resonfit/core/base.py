"""
Abstract base classes for ResonFit components.

This module defines the core interfaces that all preprocessing and fitting
components must implement.
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePreprocessor(ABC):
    """
    Base class for all preprocessing modules.
    
    A preprocessor transforms the raw S21 data to prepare it for fitting.
    Examples include cable delay correction and amplitude/phase normalization.
    """
    
    @abstractmethod
    def preprocess(self, freqs, s21):
        """
        Preprocess frequency and S21 data.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data array (Hz)
        s21 : array_like
            Complex S21 transmission data
            
        Returns
        -------
        tuple
            Processed (freqs, s21) tuple
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """
        Return the name of the preprocessor.
        
        Returns
        -------
        str
            Name of the preprocessor
        """
        pass
    
    @property
    def parameters(self):
        """
        Return the current parameters of the preprocessor.
        
        Returns
        -------
        dict
            Dictionary of parameters
        """
        return {}


class BaseFitter(ABC):
    """
    Base class for all fitting modules.
    
    A fitter analyzes preprocessed S21 data to extract resonator parameters
    such as resonance frequency, quality factors, etc.
    """
    
    @abstractmethod
    def fit(self, freqs, s21, **kwargs):
        """
        Fit frequency and S21 data.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data array (Hz)
        s21 : array_like
            Complex S21 transmission data
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        dict
            Fitting results and parameters
        """
        pass
        
    @abstractmethod
    def get_model_data(self, freqs):
        """
        Return model data for given frequencies.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data array (Hz)
            
        Returns
        -------
        array_like
            Complex model S21 data
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """
        Return the name of the fitter.
        
        Returns
        -------
        str
            Name of the fitter
        """
        pass
    
    @property
    def parameters(self):
        """
        Return the current parameters of the fitter.
        
        Returns
        -------
        dict
            Dictionary of parameters
        """
        return {}
