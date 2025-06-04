import numpy as np
from resonfit.core.models import dcm_model
import pytest
from resonfit.fitting.methods.dcm import DCMFitter

def generate_synthetic_dcm_data(freqs: np.ndarray, fr: float, Ql: float, Qc_mag: float, phi: float) -> np.ndarray:
    """
    Generates synthetic S21 data using the DCM model.

    Args:
        freqs: Numpy array of frequencies.
        fr: Resonance frequency.
        Ql: Loaded Q factor.
        Qc_mag: Magnitude of coupling Q factor.
        phi: Impedance mismatch angle in radians.

    Returns:
        Numpy array of complex S21 data.
    """
    s21_complex = dcm_model(freqs, fr, Ql, Qc_mag, phi)
    return s21_complex

def test_dcm_fitter_asymmetric_response():
    """
    Tests the DCMFitter with synthetic data from an asymmetric resonator.

    The test generates S21 data for a resonator with a known impedance
    mismatch (phi != 0), then fits this data using DCMFitter.
    The fitted parameters (fr, Ql, Qc_mag, phi) are compared against
    the true values used to generate the data.
    """
    # 1. Define true parameter values for an asymmetric resonator
    fr_true = 5.0e9  # 5 GHz
    Ql_true = 10000.0
    Qc_mag_true = 12000.0
    phi_true = 0.5  # radians, for asymmetry

    # 2. Create a frequency array
    num_points = 200
    freq_span_factor = 5
    freq_min = fr_true - freq_span_factor * (fr_true / (2 * Ql_true)) # Use f/2Q as bandwidth estimate
    freq_max = fr_true + freq_span_factor * (fr_true / (2 * Ql_true))
    freqs = np.linspace(freq_min, freq_max, num_points)

    # 3. Generate synthetic S21 data
    s21_synthetic_data = generate_synthetic_dcm_data(freqs, fr_true, Ql_true, Qc_mag_true, phi_true)

    # 4. Instantiate DCMFitter
    fitter = DCMFitter()

    # 5. Fit the data
    results = fitter.fit(freqs, s21_synthetic_data)

    # 6. Extract the fitted parameters
    fr_fit = results['fr']
    Ql_fit = results['Ql']
    Qc_mag_fit = results['Qc_mag']
    phi_fit = results['phi']

    # 7. Use pytest.approx for assertions
    assert fr_fit == pytest.approx(fr_true, rel=1e-3)
    assert Ql_fit == pytest.approx(Ql_true, rel=1e-2)
    assert Qc_mag_fit == pytest.approx(Qc_mag_true, rel=1e-2)
    assert phi_fit == pytest.approx(phi_true, rel=1e-2)
