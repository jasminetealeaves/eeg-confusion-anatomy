import pytest
import numpy as np
from src.data import load_all


def test_shapes_and_balance():
    data = load_all()
    X, y, pos = data 
    assert X.shape == (500, 4097), f"X shape is {X.shape}, expected (500, 4097)"
    assert len(y) == 500, f"y length is {len(y)}, expected 500"
    assert len(pos) == 500, f"pos length is {len(pos)}, expected 500"
    assert np.array_equal(np.bincount(y), [100, 100, 100, 100, 100])

def test_all_five_classes_present():
    data = load_all()
    X, y, pos = data 
    assert np.count_nonzero(y == 2) == 100, f"{np.count_nonzero(y == 2)} contralateral recordings found, expected 100"
    ...


