import pytest
import numpy as np
from src.data import load_all, CLASSES


@pytest.fixture(scope="module")
def data():
    # load the 500 files once and share them across every test in this file
    return load_all()

def test_shapes_and_balance(data):
    X, y, pos = data
    assert X.shape == (500, 4097), f"X shape is {X.shape}, expected (500, 4097)"
    assert len(y) == 500, f"y length is {len(y)}, expected 500"
    assert len(pos) == 500, f"pos length is {len(pos)}, expected 500"
    assert np.array_equal(np.bincount(y), [100, 100, 100, 100, 100])

def test_uppercase_txt_files_are_loaded(data):
    # set N (Contralateral) ships as .TXT, a case-sensitive glob would silently drop the whole class
    X, y, pos = data
    assert np.count_nonzero(y == 2) == 100, f"{np.count_nonzero(y == 2)} contralateral recordings found, expected 100"

# first 5 samples of file 001 in each folder, read straight from the raw files
@pytest.mark.parametrize("cls, first5", [
    (0, [12, 22, 35, 45, 69]),       # Z/Z001.txt
    (1, [-24, -22, -17, -18, -19]),  # O/O001.txt
    (2, [-42, -39, -35, -35, -36]),  # N/N001.TXT
    (3, [34, 33, 28, 22, 21]),       # F/F001.txt
    (4, [100, 124, 153, 185, 210]),  # S/S001.txt
], ids=CLASSES)
def test_first_file_of_each_class_is_first_row(data, cls, first5):
    X, y, pos = data
    idx = np.flatnonzero((y == cls) & (pos == 0))
    assert idx.tolist() == [100 * cls], f"class {cls} pos 0 is at row {idx.tolist()}, expected [{100 * cls}]"
    assert np.array_equal(X[idx[0], :5], first5), f"class {cls} starts with {X[idx[0], :5]}, expected {first5}"

def test_seizure_class_has_largest_amplitude(data):
    X, y, pos = data
    # mean absolute amplitude over all 100 recordings of each class
    amp = [abs(X[y == k]).mean() for k in range(5)]
    for k in range(4):
        assert amp[4] > 3 * amp[k], f"Seizure amplitude {amp[4]:.1f} is not over 3x class {k} ({amp[k]:.1f})"
