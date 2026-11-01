"""
Training script for ML Assignment 2 (BITS WILP, M.Tech AIML).

Dataset : Predict Students' Dropout and Academic Success (UCI id 697)
Task    : 3-class classification -> Dropout / Enrolled / Graduate

What this script does
  1. loads data/students_raw.csv and cleans up the column names
  2. makes a stratified 80/20 train/test split (test part is saved as test_data.csv)
  3. tunes and trains 5 classifiers (LR, DT, KNN, GaussianNB, SVM) with a small grid search
  4. evaluates every model on the held-out test set (Accuracy, AUC, Precision, Recall, F1, MCC)
  5. dumps the fitted pipelines + metrics into model/

Run:  python train_models.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 26  # my lucky number, also keeps the split reproducible
ROOT = Path(__file__).parent
RAW_FILE = ROOT / "data" / "students_raw.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Target"

# These columns are stored as integers in the csv but they are really codes
# (e.g. Course 9147 = Management), so treating them as numbers makes no sense
# for distance based models. They get one-hot encoded instead.
NOMINAL_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]


def load_data():
    df = pd.read_csv(RAW_FILE, sep=";")
    # one header has a stray tab ("Daytime/evening attendance\t")
    df.columns = [c.strip() for c in df.columns]
    return df


def build_preprocessor(numeric_cols):
    # rare codes (< 10 students) are lumped together, otherwise the
    # occupation columns alone blow up to ~80 dummy columns
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist",
                           min_frequency=10, sparse_output=False)
    return ColumnTransformer([
        ("cat", onehot, NOMINAL_COLS),
        ("num", StandardScaler(), numeric_cols),
    ])


def get_models():
    """model name -> (estimator, param grid). Grids are kept small on purpose."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=SEED),
            {"clf__C": [0.05, 0.1, 0.5, 1.0, 5.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"clf__max_depth": [4, 6, 8, 10, None],
             "clf__min_samples_leaf": [1, 5, 15, 30],
             "clf__criterion": ["gini", "entropy"]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 11, 15, 21, 31],
             "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2]},
        ),
        "Support Vector Machine": (
            # SVC(probability=True) is deprecated from sklearn 1.9, so the
            # probabilities (needed for AUC) come from a calibration wrapper.
            # isotonic gave a noticeably better AUC than sigmoid here (0.87 vs 0.85)
            CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED),
                                   method="isotonic", ensemble=False),
            {"clf__estimator__C": [0.5, 1, 3, 10],
             "clf__estimator__gamma": ["scale", 0.01]},
        ),
    }


def evaluate(model, X, y, classes):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    # macro averaging so the small "Enrolled" class counts as much as the others
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba, multi_class="ovr", average="macro",
                             labels=classes),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, confusion_matrix(y, y_pred, labels=classes)


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    df = load_data()
    print(f"dataset shape: {df.shape}")
    print(df[TARGET].value_counts().to_string(), "\n")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    classes = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # the test split goes into the repo so the streamlit app can be checked
    # against exactly the same rows the README numbers come from
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    print(f"train rows: {len(X_train)}, test rows: {len(X_test)} (saved test_data.csv)\n")

    numeric_cols = [c for c in X.columns if c not in NOMINAL_COLS]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    MODEL_DIR.mkdir(exist_ok=True)

    results, conf_mats, best_params = {}, {}, {}
    for name, (clf, grid) in get_models().items():
        start = time.time()
        pipe = Pipeline([("prep", build_preprocessor(numeric_cols)), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)

        best = search.best_estimator_
        scores, cm = evaluate(best, X_test, y_test, classes)
        results[name] = scores
        conf_mats[name] = cm.tolist()
        best_params[name] = {k.split("__")[-1]: v for k, v in search.best_params_.items()}

        joblib.dump(best, MODEL_DIR / file_name(name))
        print(f"{name:<24} cv-F1={search.best_score_:.3f}  "
              f"test-acc={scores['Accuracy']:.4f}  ({time.time() - start:.1f}s)  "
              f"{best_params[name]}")

    table = pd.DataFrame(results).T.round(4)
    print("\nTest set metrics:\n")
    print(table.to_string())
    table.index.name = "ML Model Name"
    table.to_csv(MODEL_DIR / "metrics.csv")

    # small metadata file the app reads (feature list, classes, model files...)
    meta = {
        "dataset": "Predict Students' Dropout and Academic Success (UCI 697)",
        "target": TARGET,
        "classes": classes,
        "features": list(X.columns),
        "nominal_features": NOMINAL_COLS,
        "random_state": SEED,
        "test_size": 0.2,
        "models": {name: file_name(name) for name in results},
        "best_params": best_params,
        "test_metrics": {n: {k: round(float(v), 4) for k, v in m.items()} for n, m in results.items()},
        "confusion_matrices": conf_mats,
    }
    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(meta, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


if __name__ == "__main__":
    main()
