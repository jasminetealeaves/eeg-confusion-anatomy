import numpy as np
import pytest
from src.data import FS, load_all
from src.features import extract, FEATURE_NAMES

t = np.arange(4097) / FS

def sine(hz, amp=1.0):
    return amp * np.sin(2 * np.pi * hz * t)

def test_shape_and_dtype():
    F = extract(np.array([sine(10), sine(20)]))
    assert F.shape == (2, len(FEATURE_NAMES)), f"F shape is {F.shape}, expected (2, {len(FEATURE_NAMES)})"
    assert F.dtype == np.float64

@pytest.mark.parametrize("hz, band_idx", [(2, 0), (6, 1), (10, 2), (20, 3), (35, 4)])
def test_pure_sine_lands_in_its_band(hz, band_idx):
    ...