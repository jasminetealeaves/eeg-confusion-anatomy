import numpy as np
import math
from src.data import load_all, FS
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}
FEATURE_NAMES = [f'rel_{b}' for b in BANDS] + ["log_total_power", "std", "line_length"]

def feature_one(seg):
    """
    get 8 features for one recording using welch's methods
    """
    freqs, psd = welch(seg, fs=FS, nperseg=512) 
    df = freqs[1] - freqs[0] # bin width 
    total_mask = (freqs >= 0.5) & (freqs < 40)
    total_power = psd[total_mask].sum() * df 

    rel = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        rel.append(psd[mask].sum() * df)

    log_total = math.log10(total_power)
    std = np.std(seg)
    line_length = np.abs(np.diff(seg)).sum()

    return [*rel, log_total, std, line_length]

def extract(X):
    """
    turn X: (n, n_samples) into (n, 8)  
    """
    all_segs_feats = []
    for seg in X:
        feats = feature_one(seg)
        all_segs_feats.append(feats)
    return np.array(all_segs_feats)


if __name__ == "__main__":
    X, y, _ = load_all()
    extract(X)