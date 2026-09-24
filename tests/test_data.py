import pytest
import numpy as np
from src.data import load_all


def test_shapes_and_balance():
    X, y, pos = load_all()
    assert X.shape == (500, 4097), f"X shape is {X.shape}, expected (500, 4097)"
    assert len(y) == 500, f"y length is {len(y)}, expected 500"
    assert len(pos) == 500, f"pos length is {len(pos)}, expected 500"
    assert np.array_equal(np.bincount(y), [100, 100, 100, 100, 100])

def test_all_five_classes_present():
    X, y, pos = load_all()
    assert np.count_nonzero(y == 2) == 100, f"{np.count_nonzero(y == 2)} contralateral recordings found, expected 100"

def test_first_file_of_each_class_is_first_row():
    X, y, pos = load_all()
    idx = np.flatnonzero((y == 3) & (pos == 0))
    assert idx.tolist() == [300]
    assert np.array_equal(X[idx[0], :5], [34, 33, 28, 22, 21])

def test_seizure_class_has_largest_amplitude():
    X, y, pos = load_all()
    cls1_x = abs(X[np.flatnonzero(y == 0)[0]]).mean()
    cls2_x = abs(X[np.flatnonzero(y == 1)[0]]).mean()
    cls3_x = abs(X[np.flatnonzero(y == 3)[0]]).mean()
    cls4_x = abs(X[np.flatnonzero(y == 4)[0]]).mean()
    assert cls4_x > 3*cls1_x, f"Seizure class recordings {cls4_x} did not exceeded 3x {cls1_x=}"
    assert cls4_x > 3*cls2_x, f"Seizure class recordings {cls4_x} did not exceeded 3x {cls2_x=}"
    assert cls4_x > 3*cls3_x, f"Seizure class recordings {cls4_x} did not exceeded 3x {cls3_x=}"