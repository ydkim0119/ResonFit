"""
Base classes for the ResonFit package.

This module defines the abstract base classes that serve as interfaces
for the different components of the resonator fitting process.
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePreprocessor(ABC):
    """
    Base class for all preprocessing modules.
    
    Preprocessors transform the raw frequency and S21 data to prepare it
    for fitting, such as removing cable delay or normalizing the data.
    """
    
    @abstractmethod
    def preprocess(self, freqs, s21):
        """
        Preprocess frequency and S21 data.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
        s21 : array_like
            Complex S21 data
            
        Returns
        -------
        tuple
            Processed (freqs, s21) tuple
        """
        pass
    
    def __str__(self):
        """String representation of the preprocessor."""
        return f"{self.__class__.__name__}"


class BaseFitter(ABC):
    """
    Base class for all fitting modules.
    
    Fitters perform the actual fitting of preprocessed S21 data to extract
    resonator parameters such as resonance frequency, quality factors, etc.
    """
    
    @abstractmethod
    def fit(self, freqs, s21, **kwargs):
        """
        Fit frequency and S21 data to extract resonator parameters.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
        s21 : array_like
            Complex S21 data
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
        Return model data for given frequencies using fitted parameters.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
            
        Returns
        -------
        array_like
            Model S21 data
        """
        pass
    
    def __str__(self):
        """String representation of the fitter."""
        return f"{self.__class__.__name__}"


class BasePlotter(ABC):
    """
    Base class for visualization modules.
    
    Plotters create visualizations of the resonator data, fitting results,
    and other relevant information.
    """
    
    @abstractmethod
    def plot(self, freqs, s21, model=None, **kwargs):
        """
        Create visualization of resonator data and model fit.
        
        Parameters
        ----------
        freqs : array_like
            Frequency data (Hz)
        s21 : array_like
            Complex S21 data
        model : array_like, optional
            Model S21 data from fitting
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        object
            Plot object (e.g., matplotlib figure)
        """
        pass
