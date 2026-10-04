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

        return np.means(f1s), np.std(f1s)

