import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.base import clone

from src.data import load_all, CLASSES
from src.features import extract, FEATURE_NAMES

def build_models():
    """
    returns dict[str, estimator]: logreg, forest, svm_rbf
    """
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000))
        ]),
        "forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale"))
        ])
    }

def random_split_eval(model, F, y, seed=42):
    # 5 folds x 5 shuffles = 25 train/test splits 
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    cm_total = np.zeros((5,5))
    f1s = []
    for train, test in cv.split(F, y):
        # train and test are indices 
        X_train = F[train]
        y_train = y[train]
        X_test = F[test]
        y_test = y[test]
        # get a fresh copy of the model so nothing gets carried over between folds
        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        # count the fold's correct and mix-ups and add to the running total across all 25 folds
        cm_total += confusion_matrix(y_test, y_pred, labels=range(5))
        # get macro f1 score. using macro f1 score here to compute f1 per class then average with equal weight 
        # this way doing well on seizure class can't dominate 
        f1s.append(f1_score(y_test, y_pred, average='macro'))

    # row normalization
    cm_total = cm_total/cm_total.sum(axis=1, keepdims=True)

    return cm_total, np.mean(f1s), np.std(f1s)

def block_split_eval(model, F, y, pos):
    # estimates how much of the random split comes from recognizing the people rather than the brain states
    # 100 recordings across 5 people, each person roughly has 20 recordings 
    f1s = []
    for b in range(5):
        test_mask = pos // 20 == b 
        train = ~test_mask
        X_train = F[train]
        y_train = y[train]
        X_test = F[test_mask]
        y_test = y[test_mask]

        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        f1s.append(f1_score(y_test, y_pred, average='macro'))

    return np.mean(f1s), np.std(f1s)

def plot(cm, title, path, vmax=1.0):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(5))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels(CLASSES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center', color="white" if cm[i, j] > 0.5*vmax else "black")
    
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()

def top_confusions(cm, k=3):
    """
    find the three biggest mistakes in the confusion matrix
    """
    off = cm.copy()
    # remove the diagonal as it holds correct predictions 
    np.fill_diagonal(off, -1)
    flat_idx = np.argsort(off, axis=None)[::-1][:k]
    rows, cols = np.unravel_index(flat_idx, off.shape)

    return [(CLASSES[i], CLASSES[j], float(cm[i,j])) for i, j in zip(rows, cols)]


def main():
    # load and extract data 
    X, y, pos = load_all()
    F = extract(X)
    print(f"{F.shape=}")
    RESULTS = Path(__file__).resolve().parents[1] / "results"
    RESULTS.mkdir(exist_ok=True)
    cms = []

    # build the models
    for name, model in build_models().items():
        cm, r_mean, r_std = random_split_eval(model, F, y)
        b_mean, b_std = block_split_eval(model, F, y, pos)
        gap = r_mean - b_mean
        print(f"{name:<8}  random {r_mean:.3f} +/- {r_std:.3f}  "
            f"block {b_mean:.3f} +/- {b_std:.3f}  gap {gap:+.3f}")
        
        # plot the results 
        title = f"{name}"
        path = RESULTS / f"cm_{name}.png"
        plot(cm, title, path)

        cms.append(cm)

    all_cms = np.stack(cms)
    cm_mean = all_cms.mean(axis=0)
    plot(cm_mean, "cm mean", RESULTS / "cm_mean.png")
    cm_std = all_cms.std(axis=0)
    plot(cm_std, "cm std", RESULTS / "cm_disagreement.png", cm_std.max())

    # check top 3 confusions among all models 
    print("Top confusions (true --> predicted):")
    for t, p, v in top_confusions(cm_mean):
        print(f"{t} --> {p} {v:.2f}")

if __name__ == "__main__": 
    main()

