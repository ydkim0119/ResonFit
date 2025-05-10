# ResonFit Development Plans

## Overview

ResonFit is a Python package designed for analyzing and fitting superconducting resonator S21 data. This document outlines our development plans for organizing the repository, implementing a modular design for extensibility, and preparing for PyPI distribution.

## Repository Structure

- [x] Create the following repository structure:

```
ResonFit/
├── LICENSE                  # MIT License
├── README.md                # Project overview and quick start guide
├── CONTRIBUTING.md          # Contribution guidelines
├── setup.py                 # Package installation configuration
├── pyproject.toml           # Build system requirements
├── resonfit/                # Main package
│   ├── __init__.py
│   ├── core/                # Core functionality
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract base classes
│   │   ├── models.py        # Model functions
│   │   └── pipeline.py      # Pipeline orchestration
│   ├── preprocessing/       # Preprocessing modules
│   │   ├── __init__.py
│   │   ├── base.py          # Base preprocessor class
│   │   ├── delay.py         # Cable delay correction
│   │   └── normalization.py # Amplitude/phase normalization
│   ├── fitting/             # Fitting modules
│   │   ├── __init__.py
│   │   ├── base.py          # Base fitter class
│   │   └── methods/         # Different fitting methods
│   │       ├── __init__.py
│   │       ├── dcm.py       # Diameter Correction Method
│   │       ├── inverse.py   # Inverse S21 Method
│   │       └── cpzm.py      # Closest Pole and Zero Method
│   └── visualization/       # Visualization tools
│       ├── __init__.py
│       └── plotter.py       # Plotting utilities
├── examples/                # Example notebooks and scripts
│   ├── basic_usage.ipynb
│   ├── custom_pipeline.ipynb
│   └── data/                # Sample data
│       └── sample_data.csv
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_fitting.py
│   └── test_visualization.py
└── docs/                    # Documentation
    ├── conf.py              # Sphinx configuration
    ├── index.rst            # Documentation index
    └── api/                 # API documentation
```

## Core Architecture Implementation

### Modular Design

- [x] Refactor current code into modular design with the following components:

#### Base Classes and Interfaces
- [x] Create `BasePreprocessor` abstract class
- [x] Create `BaseFitter` abstract class
- [x] Implement `ResonatorPipeline` class

#### Preprocessing Modules
- [x] Implement `CableDelayCorrector` class
- [x] Implement `AmplitudePhaseNormalizer` class

#### Fitting Methods
- [ ] Implement `DCMFitter` class (Diameter Correction Method) - **IN PROGRESS**
- [ ] Implement `InverseFitter` class (Inverse S21 Method)
- [ ] Implement `CPZMFitter` class (Closest Pole and Zero Method)

#### Visualization Tools
- [ ] Implement basic plotting utilities
- [ ] Implement specialized plots for each processing step

### Interface Implementations

#### BasePreprocessor

```python
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    """Base class for all preprocessing modules"""
    
    @abstractmethod
    def preprocess(self, freqs, s21):
        """
        Preprocess frequency and S21 data
        
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
        pass
```

#### BaseFitter

```python
class BaseFitter(ABC):
    """Base class for all fitting modules"""
    
    @abstractmethod
    def fit(self, freqs, s21, **kwargs):
        """
        Fit frequency and S21 data
        
        Parameters
        ----------
        freqs : array_like
            Frequency data
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
        Return model data for given frequencies
        
        Parameters
        ----------
        freqs : array_like
            Frequency data
            
        Returns
        -------
        array_like
            Model S21 data
        """
        pass
```

### Pipeline Architecture

- [x] Implement the `ResonatorPipeline` class:

```python
class ResonatorPipeline:
    """Pipeline for combining preprocessing and fitting steps"""
    
    def __init__(self):
        self.preprocessors = []
        self.fitter = None
        self.results = {}
        
    def add_preprocessor(self, preprocessor):
        """Add preprocessing step"""
        self.preprocessors.append(preprocessor)
        return self
        
    def set_fitter(self, fitter):
        """Set fitting method"""
        self.fitter = fitter
        return self
        
    def run(self, freqs, s21, plot=False):
        """Run the pipeline"""
        # Execute preprocessing steps
        processed_freqs, processed_s21 = freqs.copy(), s21.copy()
        for preprocessor in self.preprocessors:
            processed_freqs, processed_s21 = preprocessor.preprocess(processed_freqs, processed_s21)
            
        # Execute fitting
        if self.fitter:
            self.results = self.fitter.fit(processed_freqs, processed_s21)
            
        # Return results
        return self.results
```

## Implementation Roadmap

### Phase 1: Core Implementation

- [x] Refactor current code into modular structure
- [x] Implement abstract base classes and interfaces
- [ ] Implement concrete classes for existing functionality:
  - [x] CableDelayCorrector
  - [x] AmplitudePhaseNormalizer
  - [ ] DCMFitter - **IN PROGRESS**
- [x] Create basic pipeline implementation
- [ ] Implement essential visualization tools - **NEXT STEP**

### Phase 2: Testing and Documentation

- [ ] Develop comprehensive test suite
  - [ ] Unit tests for preprocessing modules
  - [ ] Unit tests for fitting methods
  - [ ] Integration tests for pipeline
- [ ] Create example notebooks
  - [ ] Basic usage example
  - [ ] Custom pipeline example
  - [ ] Advanced fitting example
- [ ] Set up documentation
  - [ ] Generate API documentation using Sphinx
  - [ ] Write detailed README
  - [ ] Create contribution guidelines

### Phase 3: PyPI Packaging

- [x] Set up package distribution
  - [x] Finalize setup.py
  - [x] Configure pyproject.toml
  - [ ] Implement version management
- [ ] Configure CI/CD
  - [ ] Set up GitHub Actions for testing
  - [ ] Set up automated documentation build
  - [ ] Configure PyPI publication workflow
- [ ] Create initial PyPI release

### Phase 4: Extension and Enhancement

- [ ] Implement advanced features
  - [ ] Add additional fitting methods
  - [ ] Create more preprocessing techniques
  - [ ] Enhance visualization capabilities
- [ ] Performance improvements
  - [ ] Optimize core algorithms
  - [ ] Add parallel processing support
  - [ ] Implement caching mechanisms

## Current Progress (May 2025)

### Completed Tasks
- Basic repository structure is set up
- Base interfaces and abstract classes are implemented
- Core module implementation is complete
- Preprocessing modules are implemented
- Basic fitting module structure is in place

### In Progress
- DCM Fitting method implementation is currently being modularized from the original `Resonator_Fitter.py` file
- Creating fitting methods for various resonator models (DCM, Inverse, CPZM)

### Next Steps
1. Complete modularization of DCMFitter from the original code
2. Implement InverseFitter and CPZMFitter classes
3. Create visualization module
4. Add examples and test cases
5. Set up CI/CD and prepare for PyPI release

## Extensibility Plan

### Adding New Preprocessing Methods

- [ ] Create examples and documentation for extending preprocessing:

```python
from resonfit.core.base import BasePreprocessor

class CustomBackgroundRemover(BasePreprocessor):
    """Custom background removal preprocessor"""
    
    def __init__(self, parameters):
        self.parameters = parameters
        
    def preprocess(self, freqs, s21):
        # Implement custom background removal
        # ...
        return freqs, s21_processed
```

### Adding New Fitting Methods

- [ ] Create examples and documentation for extending fitting methods:

```python
from resonfit.core.base import BaseFitter

class MachineLearningFitter(BaseFitter):
    """ML-based fitting method"""
    
    def __init__(self, model_params):
        self.model = None
        self.model_params = model_params
        self.fit_results = {}
        
    def fit(self, freqs, s21, **kwargs):
        # Implement ML fitting
        # ...
        return self.fit_results
        
    def get_model_data(self, freqs):
        # Generate model data
        # ...
        return model_s21
```

## Distribution Plan

### PyPI Registration

- [ ] Prepare for PyPI distribution
  - [x] Choose package name (resonfit)
  - [ ] Create PyPI account
  - [ ] Set up distribution workflow:
    ```bash
    python -m pip install --upgrade build twine
    python -m build
    python -m twine upload dist/*
    ```

### Versioning Strategy

- [ ] Implement Semantic Versioning (SemVer):
  - MAJOR version for incompatible API changes
  - MINOR version for new functionality in a backward-compatible manner
  - PATCH version for backward-compatible bug fixes

### CI/CD Pipeline

- [ ] Set up GitHub Actions workflow for:
  - [ ] Running tests
  - [ ] Building documentation
  - [ ] Publishing to PyPI

## Future Enhancements

### Planned Features

- [ ] Interactive Visualization
  - [ ] Implement plots using Plotly or Bokeh
  - [ ] Create interactive parameter adjustment
- [ ] Advanced Fitting Methods
  - [ ] Add Bayesian fitting methods
  - [ ] Implement machine learning approaches
- [ ] Automated Parameter Selection
  - [ ] Create intelligent initial guess algorithms
  - [ ] Implement adaptive weighting schemes
- [ ] Batch Processing
  - [ ] Add support for multiple dataset analysis
  - [ ] Implement parallel processing for batch jobs
- [ ] GUI Interface (long-term)
  - [ ] Design simple graphical interface
  - [ ] Create standalone application

### Community Contribution Areas

- [ ] Document contribution opportunities
  - [ ] Additional fitting methods
  - [ ] Support for different data formats
  - [ ] Integration with other scientific Python packages
  - [ ] Performance optimizations
  - [ ] Extended documentation and tutorials

## Getting Started for Contributors

- [ ] Create contributor documentation with:
  - [ ] Development environment setup instructions
  - [ ] Coding standards
  - [ ] Pull request process
  - [ ] Testing requirements

```bash
# Setup instructions
git clone https://github.com/ydkim0119/ResonFit.git
cd ResonFit
pip install -e ".[dev]"

# Creating a feature branch
git checkout -b feature/your-feature-name

# Run tests
pytest
```

We welcome contributions from the community to help make ResonFit a comprehensive and robust tool for superconducting resonator analysis!