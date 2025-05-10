# ResonFit

A modular Python package for analyzing and fitting superconducting resonator S21 data.

## Overview

ResonFit provides a comprehensive and extensible framework for analyzing S21 transmission data from superconducting microwave resonators. It is designed to be modular and flexible, allowing users to customize their analysis pipeline while providing sensible defaults based on established methods.

The package is currently under active development. See our [Development Plan](DEVELOPMENT_PLAN.md) for the current implementation status and future plans.

## Features (Planned)

- 📊 **Data Preprocessing**
  - Cable delay correction
  - Amplitude and phase normalization
  - Background removal

- 🔍 **Fitting Methods**
  - DCM (Diameter Correction Method)
  - Inverse S21 Method
  - CPZM (Closest Pole and Zero Method)

- 🧩 **Flexible Pipeline**
  - Combine different preprocessing and fitting methods
  - Customize each step of the analysis process

- 📈 **Visualization Tools**
  - Complex plane plots
  - Amplitude and phase plots
  - Residual analysis

## Installation (Coming Soon)

```bash
pip install resonfit
```

## Quick Example (Preview)

```python
from resonfit import ResonatorPipeline
from resonfit.preprocessing import CableDelayCorrector, AmplitudePhaseNormalizer
from resonfit.fitting.methods import DCMFitter
import numpy as np

# Load data
freqs = np.linspace(5e9, 6e9, 1001)
s21 = your_data_loading_function()

# Create pipeline
pipeline = ResonatorPipeline()
pipeline.add_preprocessor(CableDelayCorrector())
pipeline.add_preprocessor(AmplitudePhaseNormalizer())
pipeline.set_fitter(DCMFitter(use_weights=True))

# Run fitting
results = pipeline.run(freqs, s21, plot=True)

# Print results
print(f"Resonance frequency: {results['fr']/1e9:.6f} GHz")
print(f"Quality factors: Qi={results['Qi']:.0f}, Qc={results['Qc_mag']:.0f}, Ql={results['Ql']:.0f}")
```

## Development Status

ResonFit is currently in the early development stage. You can track our progress in the [Development Plan](DEVELOPMENT_PLAN.md).

## Contributing

Contributions are welcome! If you'd like to help with the development of ResonFit, please check our [Development Plan](DEVELOPMENT_PLAN.md) for areas that need attention.

## License

This project will be licensed under the MIT License (to be added).