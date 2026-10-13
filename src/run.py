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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.base import clone

from src.data import load_all, CLASSES
from src.features import extract, extract_spectrum

def build_models():
    """
    returns dict[str, estimator]: logistic regression, random forest, svm_rbf
    """
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return {
        "logreg": GridSearchCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000))
            ]),
            param_grid={"clf__C": [0.01, 0.1, 1, 10, 100]},
            cv=inner_cv, scoring="f1_macro", n_jobs=-1
        ),
        "forest": GridSearchCV(
            RandomForestClassifier(n_estimators=300, random_state=42),
            param_grid={"min_samples_leaf": [1, 3, 10], "max_features": ["sqrt", 0.3]},
            cv=inner_cv, scoring="f1_macro", n_jobs=-1
        ),
        "svm_rbf": GridSearchCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf"))
            ]),
            param_grid={"clf__C": [0.1, 1, 10, 100, 1000], "clf__gamma": ["scale", 0.001, 0.01, 0.1]},
            cv=inner_cv, scoring="f1_macro", n_jobs=-1
        )
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
    # sort in descending order
    flat_idx = np.argsort(off, axis=None)[::-1][:k]
    rows, cols = np.unravel_index(flat_idx, off.shape)

    return [(CLASSES[i], CLASSES[j], float(cm[i,j])) for i, j in zip(rows, cols)]


def main():
    # load and extract data 
    X, y, pos = load_all()
    feature_sets = {"bands": extract(X), "spectrum": extract_spectrum(X)}
    RESULTS = Path(__file__).resolve().parents[1] / "results"
    RESULTS.mkdir(exist_ok=True)

    # build the models
    for set_name, F in feature_sets.items():
        print("-"*100)
        print(f"Now examining {set_name}:")
        cms = []
        for name, model in build_models().items():
            cm, r_mean, r_std = random_split_eval(model, F, y)
            # check performance without grid search
            cm_u, u_mean, _ = random_split_eval(model.estimator, F, y)
            print(f"{name:<8}  tuned F1 {r_mean:.3f} +/- {r_std:.3f}  untuned F1 {u_mean:.3f}")
            # plot the results 
            title = f"{set_name} {name}"
            path = RESULTS / f"cm_{set_name}_{name}.png"
            plot(cm, title, path)

            cms.append(cm)

        all_cms = np.stack(cms)
        cm_mean = all_cms.mean(axis=0)
        plot(cm_mean, f"Confusion Matrix Mean ({set_name})", RESULTS / f"cm_{set_name}_mean.png")
        cm_std = all_cms.std(axis=0)
        plot(cm_std, f"Confusion Matrix Standard Deviation ({set_name})", RESULTS / f"cm_{set_name}_disagreement.png", cm_std.max())

        # check top 3 confusions among all models 
        print(f"Top confusions for {set_name} (true --> predicted):")
        for t, p, v in top_confusions(cm_mean):
            print(f"{t} --> {p} {v:.2f}")

if __name__ == "__main__": 
    main()

