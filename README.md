# EEG Confusion Anatomy

This project classifies 5 kinds of neural recordings from the Bonn EEG Epileptic dataset (Andrzejak et al., 2001) to see which classes the models confuse.

| Class | Who | Electrode | Condition |
| --- | --- | --- | --- |
| **EyeOpen** | 5 healthy volunteers | on the scalp | resting, eyes open |
| **EyeClosed** | same 5 healthy | on the scalp | resting, eyes closed |
| **Contralateral** | 5 epilepsy patients | inside the skull | between seizures, from the *non*-epileptogenic side of the brain |
| **Epileptogenic** | same 5 epilepsy patients | inside the skull | between seizures, from the epileptogenic side |
| **Seizure** | same 5 epilepsy patients | inside the skull | during an actual seizure |

Detecting a seizure is easy: its amplitude is roughly 5 times that of the other classes. This project focuses on Contralateral vs Epileptogenic: can a model tell apart EEG recorded from the same patients, with the same type of depth electrode placed at matching spots in each hippocampus, when one side is epileptogenic tissue and the other is not?

## Findings

- **Short answer: partly.** Epileptogenic recordings are correctly identified 62% of the time with band features and 75% with the full spectrum. Most misses are called Contralateral (32% / 21%).
- Epileptogenic ↔ Contralateral is the most confused pair for every model and both feature sets.

**Top confusions, averaged across the 3 models**

| True → Predicted | Bands | Spectrum |
| --- | --- | --- |
| Epileptogenic → Contralateral | 0.32 | 0.21 |
| Contralateral → Epileptogenic | 0.18 | 0.16 |
| EyeClosed → EyeOpen | 0.07 | 0.05 |

<p>
  <img src="results/cm_bands_mean.png" alt="Average confusion matrix, band features" width="49%">
  <img src="results/cm_spectrum_mean.png" alt="Average confusion matrix, full spectrum" width="49%">
</p>

- **The full spectrum helps logistic regression and the SVM, but not the forest.** Epileptogenic → Contralateral falls from 0.34 to 0.16 for logistic regression and from 0.36 to 0.17 for the SVM, but goes from 0.27 to 0.29 for the forest (per-model matrices are in `results/`).
- **Why Epileptogenic → Contralateral is more common than the reverse.** A likely explanation: the paper says interictal epileptiform activity (spikes and sharp waves) comes and goes, so only some Epileptogenic segments contain it. It found nonlinear structure in 37/100 Epileptogenic vs 14/100 Contralateral segments. Normal-looking Epileptogenic segments fall inside the Contralateral range, where Contralateral recordings outnumber them, so the models call them Contralateral. Contralateral recordings rarely look like the abnormal Epileptogenic ones, so the reverse mistake is rarer. The gap is clearest with band features (0.32 vs 0.18). The full spectrum mainly cuts the Epileptogenic → Contralateral errors (0.21 vs 0.16), and for logistic regression the gap disappears (0.16 vs 0.17).

**Macro F1 (mean ± std across folds), averaged over all 5 classes**

| Features | Model | Tuned | Untuned (scikit-learn defaults) |
| --- | --- | --- | --- |
| Bands | logreg | 0.815 ± 0.032 | 0.798 |
| Bands | forest | 0.858 ± 0.023 | 0.859 |
| Bands | svm_rbf | 0.847 ± 0.032 | 0.808 |
| Spectrum | logreg | 0.882 ± 0.031 | 0.877 |
| Spectrum | forest | 0.865 ± 0.031 | 0.865 |
| Spectrum | svm_rbf | 0.912 ± 0.028 | 0.824 |

Tuning matters mostly for the SVM. It changes how often Epileptogenic is called Contralateral (average of the 3 models: 0.36 → 0.32 with bands, 0.31 → 0.21 with the spectrum) but not which classes are confused: the same three pairs top the list.

## What this can and can't tell us

- The data was filtered at 0.53-40 Hz and sampled at 173.61 Hz. High frequency oscillations (80-500 Hz), a key marker of epileptogenic tissue, cannot appear in the recordings.
- Patient recognition can't be ruled out. There are no patient IDs, and the random-split scores may be optimistic.
- Healthy volunteers were recorded from the scalp and patients from inside the skull, so telling healthy and patient classes apart may partly reflect electrode type rather than brain state. This also inflates the 5-class macro F1. The Contralateral vs Epileptogenic comparison isn't affected, as both use the same type of depth electrode.

## Method

**Data.**
500 single-channel recordings, 100 per class. Each is 23.6 s long: 4097 samples at 173.61 Hz, band-pass filtered at 0.53–40 Hz. `src/data.py` loads them into `X` (500 × 4097), the labels `y` (0–4), and `pos`, each recording's position (0–99) within its class. See [data/README.md](data/README.md) for download steps and file layout.

**Features** (`src/features.py`). Both feature sets come from Welch's power spectrum (512-sample windows, bins 0.339 Hz apart):

| Feature set | Values per recording | Contents |
| --- | --- | --- |
| Bands | 8 | relative power in delta (0.5–4 Hz), theta (4–8), alpha (8–13), beta (13–30) and gamma (30–40); log10 total power (0.5–40 Hz); standard deviation; line length |
| Spectrum | 116 | log10 power at every bin from 0.5 to 40 Hz |

The band powers are a summary of the full spectrum, so comparing the two sets shows how much of the confusion comes from summarizing.

**Models** (`src/run.py`), all from scikit-learn:

| Model | Pipeline | Settings searched |
| --- | --- | --- |
| logistic regression | `StandardScaler` + `LogisticRegression` | `C`: 0.01, 0.1, 1, 10, 100 |
| random forest | `RandomForestClassifier` (300 trees) | `min_samples_leaf`: 1, 3, 10; `max_features`: sqrt, 0.3 |
| SVM with an RBF kernel | `StandardScaler` + `SVC` | `C`: 0.1, 1, 10, 100, 1000; `gamma`: scale, 0.001, 0.01, 0.1 |

The scalers are part of each model's pipeline, so they are fit on the training folds only.

**Evaluation.**

- **Random split:** stratified 5-fold cross-validation, repeated 5 times, for 25 train/test splits in total. It reports macro F1 (mean ± std over folds) and a confusion matrix summed over all folds, with each row divided by its total.
- **Hyperparameter tuning:** within each of the 25 training sets, a 5-fold grid search picks the settings with the best macro F1, and the model is refit on the whole training set before scoring the test fold (nested cross-validation). The test fold never influences the choice. For comparison, the same models with scikit-learn's default settings are scored on the same 25 splits.
- **Model agreement:** the three models' confusion matrices are averaged (`cm_<set>_mean.png`), and their standard deviation is plotted (`cm_<set>_disagreement.png`).

## Reproduce

Run all commands from the repository root.

1. Download the data into `data/raw/Dataset/{Z,O,N,F,S}/` as described in [data/README.md](data/README.md).
2. Create the environment with Python 3.12 and [uv](https://docs.astral.sh/uv/):

   ```bash
   uv venv --python 3.12
   uv pip install -r requirements.txt
   ```

3. Run the tests:

   ```bash
   uv run python -m pytest tests/ -v
   ```

4. Run the experiment:

   ```bash
   uv run python -m src.run
   ```

   It takes about 3 minutes. For each feature set it prints tuned and untuned macro F1 per model, the Epileptogenic → Contralateral rate with and without tuning, and the top confusions. It saves these plots to `results/`:

   - `cm_<set>_<model>.png`: confusion matrix for one model
   - `cm_<set>_mean.png`: average over the 3 models
   - `cm_<set>_disagreement.png`: standard deviation across the 3 models

   `<set>` is `bands` or `spectrum`, and `<model>` is `logreg`, `forest` or `svm_rbf`. Random seeds and package versions are fixed, so reruns give the same numbers.

## Reference

- R. G. Andrzejak, K. Lehnertz, F. Mormann, C. Rieke, P. David, C. E. Elger. *Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state.* Phys. Rev. E **64**, 061907 (2001). <https://doi.org/10.1103/PhysRevE.64.061907>
- Dataset: official copy at <https://doi.org/10.34810/data490>. This project uses the Kaggle mirror at <https://www.kaggle.com/datasets/quands/eeg-dataset>.
