import numpy as np
import pytest
from src.data import FS, load_all
from src.features import extract, extract_spectrum, FEATURE_NAMES

t = np.arange(4097) / FS

# frequency of each spectrum column: the bins welch returns for nperseg=512, kept between 0.5 and 40 Hz
_all_freqs = np.fft.rfftfreq(512, d=1/FS)
SPEC_FREQS = _all_freqs[(_all_freqs >= 0.5) & (_all_freqs < 40)]

def sine(hz, amp=1.0):
    # standard formula for a sine wave: signal(t) = A · sin(2π · f · t)
    return amp * np.sin(2 * np.pi * hz * t)

def test_spectrum_shape():
    S = extract_spectrum(np.array([sine(10), sine(20)]))
    assert S.shape == (2, 116), f"S shape is {S.shape}, expected (2, 116)"

@pytest.mark.parametrize("hz, band_idx", [(2, 0), (6, 1), (10, 2), (20, 3), (35, 4)])
def test_pure_sine_lands_in_its_band(hz, band_idx):
    # 'None' adds a new axis of length 1. extract function needs a 2D array like X (n_recordings, n_samples)
    # get the 5 relative band powers in row 0 (row 0, col 0-4)
    rel = extract(sine(hz)[None])[0, :5] 
    assert rel[band_idx] > 0.95, f"Pure sine wave does not land in its own band. {rel[band_idx]=}"

def test_rel_bands_sum_to_one():
    rng = np.random.default_rng(0)
    rel = extract(rng.normal(size=(3,4097)))[:, :5]
    assert np.allclose(rel.sum(axis=1), 1), f"Rel bands do not sum to one, {rel.sum(axis=1)=}"

def test_shape_and_dtype():
    F = extract(np.array([sine(10), sine(20)]))
    assert F.shape == (2, len(FEATURE_NAMES)), f"F shape is {F.shape}, expected (2, {len(FEATURE_NAMES)})"

def test_spectrum_peaks_at_sine_frequency():
    S = extract_spectrum(sine(10)[None])[0]
    peak = SPEC_FREQS[np.argmax(S)]
    assert abs(peak - 10) < 0.5, f"peak at {peak} Hz, expected ~10 Hz"