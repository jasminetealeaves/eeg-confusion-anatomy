import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Dataset"
N_SAMPLES = 4097
SETS = ["Z", "O", "N", "F", "S"]

def get_files_per_folder(s):
    folder = ROOT / s
    files = [file for file in folder.glob("*.txt", case_sensitive=False)]
    return sorted(files, key=lambda p:int(p.stem[1:]))

def load_all():
    """
    This function loads the txt files. 
    """
    X, y, pos = [], [], []
    for cls, s in enumerate(SETS):
        files = get_files_per_folder(s)
        if len(files) != 100:
            raise ValueError(f'{set=} found {len(files)}, expected 100')
        for i, file in enumerate(files):
            data = np.loadtxt(file, dtype=np.float64)
            if data.shape != (N_SAMPLES,):
                raise ValueError(f'{file.name}: shape is {data.shape}, expected ({N_SAMPLES},)')
            X.append(data)
            y.append(cls)
            pos.append(i)
    return np.array(X), np.array(y), np.array(pos)
    


if __name__ == "__main__":
    load_all()