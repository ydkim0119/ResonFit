# ResonFit Development Plans

## Overview

ResonFit is a Python package designed for analyzing and fitting superconducting resonator S21 data. This document outlines our development plans for organizing the repository, implementing a modular design for extensibility, and preparing for PyPI distribution.

## Repository Structure

- [x] Create the following repository structure:

```
ResonFit/
├── LICENSE                  # MIT License
├── README.md                # Project overview and quick start guide
├── CONTRIBUTING.md          # Contribution guidelines (To be created)
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
│   │       ├── inverse.py   # Inverse S21 Method (Planned)
│   │       └── cpzm.py      # Closest Pole and Zero Method (Planned)
│   └── visualization/       # Visualization tools
│       ├── __init__.py
│       └── plotter.py       # Plotting utilities
├── examples/                # Example notebooks and scripts
│   ├── basic_usage.py       # Marimo example for basic usage
│   ├── resonator_fitter_legacy.ipynb # Legacy comparison
│   └── data/                # Sample data (Planned)
│       └── sample_data.csv  # (Planned)
├── tests/                   # Test suite (To be developed)
│   ├── __init__.py
│   ├── test_preprocessing.py # (Planned)
│   ├── test_fitting.py       # (Planned)
│   └── test_visualization.py # (Planned)
└── docs/                    # Documentation (To be developed)
    ├── conf.py              # Sphinx configuration (Planned)
    ├── index.rst            # Documentation index (Planned)
    └── api/                 # API documentation (Planned)
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
- [x] Implement `DCMFitter` class (Diameter Correction Method)
- [ ] Implement `InverseFitter` class (Inverse S21 Method) - **NEXT STEP**
- [ ] Implement `CPZMFitter` class (Closest Pole and Zero Method)

#### Visualization Tools
- [x] Implement basic plotting utilities in `ResonancePlotter`
- [x] Implement specialized plots for each processing step accessible via `ResonancePlotter` and pipeline's `run_analysis_and_plot`

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
from abc import ABC, abstractmethod # Added import for completeness

class BaseFitter(ABC): # Added ABC inheritance
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
    def get_model_data(self, freqs): # freqs added as argument
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
- [x] Add `run_analysis_and_plot` method for single-call full analysis and plotting.

```python
class ResonatorPipeline:
    """Pipeline for combining preprocessing and fitting steps"""
    
    def __init__(self):
        self.preprocessors = []
        self.fitter = None
        self.results = {}
        self._intermediate_results = {}
        self.plotter = None # Added for run_analysis_and_plot
        
    def add_preprocessor(self, preprocessor):
        """Add preprocessing step"""
        self.preprocessors.append(preprocessor)
        return self
        
    def set_fitter(self, fitter):
        """Set fitting method"""
        self.fitter = fitter
        return self
        
    def run(self, freqs, s21, plot=False): # Simplified for brevity, see actual implementation
        """Run the pipeline"""
        # ... (implementation for running steps) ...
        return self.results

    def run_analysis_and_plot(self, freqs, s21): # New method
        """Run the full analysis pipeline and generate plots for each significant step."""
        # ... (implementation for running steps and plotting) ...
        return self.results
```

## Implementation Roadmap

### Phase 1: Core Implementation & Basic Functionality (Largely Complete)

- [x] Refactor current code into modular structure
- [x] Implement abstract base classes and interfaces
- [x] Implement concrete classes for existing functionality:
  - [x] `CableDelayCorrector`
  - [x] `AmplitudePhaseNormalizer`
  - [x] `DCMFitter`
  - [x] `ResonancePlotter` with detailed plotting methods
- [x] Create `ResonatorPipeline` with basic `run` and comprehensive `run_analysis_and_plot`
- [x] Create basic example (`basic_usage.py` Marimo app)

### Phase 2: Complete the Fitting Methods Implementation

- [ ] Implement remaining fitting methods:
  - [ ] `InverseFitter` - **NEXT STEP**
  - [ ] `CPZMFitter`
- [ ] Add examples for each new fitting method
- [ ] Implement fitting method comparison functionality (Planned)
- [ ] Add Q-factor vs. power analysis utilities more deeply into the library (Conceptual in example now)

### Phase 3: Testing and Documentation

- [ ] Develop comprehensive test suite
  - [ ] Unit tests for preprocessing modules
  - [ ] Unit tests for fitting methods
  - [ ] Integration tests for pipeline
- [ ] Create/Update example notebooks/scripts
  - [x] Basic usage example (`basic_usage.py` Marimo app)
  - [ ] Custom pipeline example (Can be derived from basic usage)
  - [ ] Advanced fitting example (Once new fitters are added)
  - [ ] Fitting method comparison example (Planned)
- [ ] Set up documentation
  - [ ] Generate API documentation using Sphinx
  - [ ] Write detailed README (Good start, can be expanded)
  - [ ] Create contribution guidelines (`CONTRIBUTING.md`)

### Phase 4: PyPI Packaging & Distribution (Partially Complete)

- [x] Set up package distribution files
  - [x] Finalize `setup.py`
  - [x] Configure `pyproject.toml`
- [x] Implement version management (Manual updates to `setup.py` `version` for now)
- [x] Perform initial PyPI release (Manual steps completed/guided)
- [ ] Configure CI/CD
  - [ ] Set up GitHub Actions for automated testing on push/PR
  - [ ] Set up automated documentation build and deployment (e.g., ReadTheDocs)
  - [ ] Configure automated PyPI publication on new Git tags/releases (Planned)

### Phase 5: Extension and Enhancement (Future)

- [ ] Implement advanced features
  - [ ] Add additional fitting methods (e.g., Bayesian)
  - [ ] Create more preprocessing techniques (e.g., background subtraction)
  - [ ] Enhance visualization capabilities (e.g., interactive plots with Plotly/Bokeh)
- [ ] Performance improvements
  - [ ] Optimize core algorithms
  - [ ] Add parallel processing support for batch analysis (if applicable)
- [ ] GUI Interface (long-term consideration)

## Current Progress (As of Update)

### Completed Tasks
- [x] Basic repository structure is set up.
- [x] Base interfaces and abstract classes are implemented.
- [x] Core module implementation for preprocessing, DCM fitting, and visualization is complete.
- [x] `ResonatorPipeline` with `run` and `run_analysis_and_plot` methods implemented.
- [x] `ResonancePlotter` provides detailed plots for each analysis stage.
- [x] `basic_usage.py` Marimo example demonstrates full analysis and plotting.
- [x] Package setup configuration (`setup.py`, `pyproject.toml`) is complete for PyPI.
- [x] Guidance for manual PyPI release provided.

### In Progress
- [ ] Implementing `InverseFitter` class - **NEXT MAJOR STEP**
- [ ] Implementing `CPZMFitter` class
- [ ] Developing a comprehensive test suite.
- [ ] Setting up formal documentation (Sphinx).
- [ ] Creating `CONTRIBUTING.md`.

### Next Steps
1.  Complete the implementation of `InverseFitter` class.
2.  Implement `CPZMFitter` class.
3.  Develop unit and integration tests for all modules.
4.  Create `CONTRIBUTING.md` and formal Sphinx documentation.
5.  Set up CI/CD for automated testing and (optionally) PyPI deployment.
6.  Consider community feedback and plan for Phase 5 enhancements.

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
        return freqs_processed, s21_processed # Ensure correct return names
```

### Adding New Fitting Methods

- [ ] Create examples and documentation for extending fitting methods:

```python
from resonfit.core.base import BaseFitter
import numpy as np # Added for model_s21 example

class MachineLearningFitter(BaseFitter):
    """ML-based fitting method"""
    
    def __init__(self, model_params):
        self.model = None # e.g. a pre-trained ML model
        self.model_params = model_params
        self.fit_results = {}
        
    def fit(self, freqs, s21, **kwargs):
        # Implement ML fitting
        # e.g., self.model.predict(...)
        # Populate self.fit_results with standard keys like 'fr', 'Ql', etc.
        # ...
        self.fit_results = {"fr": 0.0, "Ql": 0.0} # Example
        return self.fit_results
        
    def get_model_data(self, freqs):
        # Generate model S21 data based on fitted parameters
        # This might involve using self.fit_results to reconstruct the S21 curve
        # ...
        model_s21 = np.ones_like(freqs) # Placeholder
        return model_s21
```

## Distribution Plan

### PyPI Registration

- [x] Prepare for PyPI distribution:
  - [x] Choose package name (`resonfit`).
  - [x] Create PyPI account and API Token.
  - [x] Set up distribution workflow (manual steps performed):
    ```bash
    python -m pip install --upgrade build twine
    python -m build
    # Optional: python -m twine upload --repository testpypi dist/* (for testing)
    python -m twine upload dist/*
    ```

### Versioning Strategy

- [x] Implement Semantic Versioning (SemVer):
  - MAJOR version for incompatible API changes.
  - MINOR version for new functionality in a backward-compatible manner.
  - PATCH version for backward-compatible bug fixes.
  - (Currently at `0.1.0` as per `setup.py`)

### CI/CD Pipeline (Planned)

- [ ] Set up GitHub Actions workflow for:
  - [ ] Running tests on push and pull requests.
  - [ ] Building documentation automatically.
  - [ ] Publishing to PyPI automatically on new Git tags/releases.

## Future Enhancements (Planned Features from original)

- [ ] Interactive Visualization (e.g., Plotly/Bokeh integration)
- [ ] Advanced Fitting Methods (e.g., Bayesian, other ML approaches)
- [ ] Automated Parameter Selection (intelligent initial guesses, adaptive weighting)
- [ ] Batch Processing and Parallelization
- [ ] GUI Interface (long-term)

## Getting Started for Contributors

- [ ] Create `CONTRIBUTING.md` with:
  - [ ] Development environment setup instructions.
  - [ ] Coding standards (e.g., Black, isort - already in `pyproject.toml`).
  - [ ] Pull request process.
  - [ ] Testing requirements.

Example setup for contributors (to be formalized in `CONTRIBUTING.md`):
```bash
# Clone the repository
git clone https://github.com/ydkim0119/ResonFit.git # Update with actual URL if different
cd ResonFit

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"

# Creating a feature branch
git checkout -b feature/your-feature-name

# Run tests (once tests are added)
pytest
```

We welcome contributions from the community to help make ResonFit a comprehensive and robust tool for superconducting resonator analysis!