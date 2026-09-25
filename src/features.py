import numpy as np
from src.data import load_all
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}
FEATURE_NAMES = [f'rel_{b}' for b in BANDS] + ["log_total_power", "std", "line_length"]
FS = 173.61

def feature_one(seg):
    """
    get 8 features for one recording using welch's methods
    """
    freqs, psd = welch(seg, fs=FS, nperseg=512) 
    df = freqs[1] - freqs[0] # power spectra density at each frequency 
    ...

if __name__ == "__main__":
    X, _, _ = load_all()
    seg = X[0]
    feature_one(seg)