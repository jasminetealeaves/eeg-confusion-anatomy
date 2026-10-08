# Data

Bonn EEG dataset (Andrzejak et al., 2001). It has 500 single-channel recordings
in 5 sets of 100. Each recording is 23.6 s long: 4097 samples at **173.61 Hz**,
band-pass filtered at 0.53–40 Hz.

Data needs to be fetched by following the steps below.

## Layout

```text
data/
  README.md
  PhysRevE.64.061907.pdf      source paper (gitignored, *.pdf)
  raw/                        gitignored
    eeg-dataset.zip           Kaggle quands/eeg-dataset, 500 files
    Dataset/                  unzipped; this is what src/data.py reads
      Z/Z001.txt … Z100.txt
      O/O001.txt … O100.txt
      N/N001.TXT … N100.TXT
      F/F001.txt … F100.txt
      S/S001.txt … S100.txt
```

Each `.txt` file holds one recording: 4097 integers, one per line, no header,
CRLF line endings.

## Sets

| Folder | Paper set | Index | Name | Subjects | Electrodes | State |
| --- | --- | --- | --- | --- | --- | --- |
| `Z` | A | 0 | EyesOpen | 5 healthy | scalp | awake, eyes open |
| `O` | B | 1 | EyesClosed | same 5 healthy | scalp | awake, eyes closed |
| `N` | C | 2 | Contralateral | 5 epilepsy patients | intracranial | interictal, opposite hemisphere |
| `F` | D | 3 | Epileptogenic | same 5 patients | intracranial | interictal, epileptogenic zone |
| `S` | E | 4 | Seizure | same 5 patients | intracranial | ictal |

`Index` is the label that `src/data.py` assigns (`SETS = ["Z", "O", "N", "F", "S"]`).
Within each set, files are ordered by their numeric suffix, which becomes `pos` (0–99).

## Download

This repo uses a Kaggle mirror of the original archive:
**<https://www.kaggle.com/datasets/quands/eeg-dataset>**

1. Download it and save it as `data/raw/eeg-dataset.zip`.
2. Unzip it inside `data/raw/`:

   ```bash
   cd data/raw && unzip eeg-dataset.zip
   ```

   This creates `data/raw/Dataset/{Z,O,N,F,S}/`, with 100 files in each folder.
3. Check the result from the repo root:

   ```bash
   uv run python -m pytest tests/test_data.py -v
   ```

   Sanity anchor: `F/F001.txt` starts with `34, 33, 28, 22, 21`.

**Official source:** <https://doi.org/10.34810/data490> (CSUC Dataverse). It
offers the same five sets as separate archives. If you use it, unzip them into
the same `Dataset/{Z,O,N,F,S}/` layout.

## Citation

> R. G. Andrzejak, K. Lehnertz, F. Mormann, C. Rieke, P. David, C. E. Elger.
> *Indications of nonlinear deterministic and finite-dimensional structures in
> time series of brain electrical activity: Dependence on recording region and
> brain state.* Phys. Rev. E **64**, 061907 (2001).
> <https://doi.org/10.1103/PhysRevE.64.061907>

```bibtex
@article{andrzejak2001,
  author  = {Andrzejak, Ralph G. and Lehnertz, Klaus and Mormann, Florian and
             Rieke, Christoph and David, Peter and Elger, Christian E.},
  title   = {Indications of nonlinear deterministic and finite-dimensional
             structures in time series of brain electrical activity:
             Dependence on recording region and brain state},
  journal = {Physical Review E},
  volume  = {64},
  number  = {6},
  pages   = {061907},
  year    = {2001},
  doi     = {10.1103/PhysRevE.64.061907}
}
```
