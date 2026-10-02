"""Run the reproducible experiment: python -m chorus_hit.train."""
import argparse
from pathlib import Path
import os
import tempfile
import json
import platform
import time
import warnings
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .config import FEATURE_COLUMNS, MODEL_PATH, RESULTS, SEED
from .data import data_audit, load_data, split_data


def pipeline(model, use_pca=True):
    steps = [("imputer", SimpleImputer(strategy="median")),
             ("variance", VarianceThreshold()), ("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    return Pipeline([*steps, ("model", model)])


def candidates():
    return {
        "Majority baseline": (pipeline(DummyClassifier(strategy="prior"), False), {}),
        "Logistic regression": (pipeline(LogisticRegression(max_iter=3000, random_state=SEED)),
                                {"model__C": [0.01, 0.1, 1.0]}),
        "LDA": (pipeline(LinearDiscriminantAnalysis(solver="lsqr")),
                {"model__shrinkage": ["auto", 0.3, 0.7]}),
        "Linear SVM": (pipeline(SVC(kernel="linear", random_state=SEED)),
                       {"model__C": [0.01, 0.1, 1.0]}),
        "RBF SVM": (pipeline(SVC(kernel="rbf", random_state=SEED)),
                    {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.001]}),
        "Polynomial SVM": (pipeline(SVC(kernel="poly", degree=2, random_state=SEED)),
                           {"model__C": [0.1, 1.0, 10.0]}),
        "Random forest": (pipeline(RandomForestClassifier(n_estimators=250, random_state=SEED,
                                                         n_jobs=1), False),
                          {"model__max_depth": [None, 8], "model__min_samples_leaf": [2, 5]}),
        "Gradient boosting": (pipeline(GradientBoostingClassifier(n_estimators=100,
                                                                 random_state=SEED), False),
                              {"model__learning_rate": [0.03, 0.1], "model__max_depth": [1, 2]}),
        "Neural network": (pipeline(MLPClassifier(hidden_layer_sizes=(64, 32), solver="adam",
                                                  max_iter=800, early_stopping=False,
                                                  random_state=SEED, tol=1e-4)),
                           {"model__alpha": [1.0, 10.0]}),
    }


def model_scores(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X), "decision score (threshold 0)"
    positive = list(model.classes_).index(1)
    return model.predict_proba(X)[:, positive], "uncalibrated class-1 score (threshold 0.5)"


def metrics(y, predictions, scores):
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def artist_bootstrap(y, predictions, scores, groups, repeats=2000):
    """Resample whole test artists, preserving within-artist dependence."""
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    samples = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == artist) for artist in sampled])
        if len(np.unique(y[indices])) < 2:
            continue
        samples.append(metrics(y[indices], predictions[indices], scores[indices]))
    draws = pd.DataFrame(samples)
    return {column: [float(x) for x in draws[column].quantile([0.025, 0.975])]
            for column in draws.columns}


def main():
    global RESULTS, MODEL_PATH
    parser = argparse.ArgumentParser(description="Reproduce V1 into a NEW directory; baseline is never overwritten")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        parser.error("Output directory must not exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".v1-reproduction-", dir=destination.parent))
    RESULTS = staging / "results"
    MODEL_PATH = staging / "models" / "selected_model.joblib"
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = load_data()
    train_idx, test_idx, groups = split_data(frame)
    X_train, X_test = frame.iloc[train_idx][FEATURE_COLUMNS], frame.iloc[test_idx][FEATURE_COLUMNS]
    y_train, y_test = frame.label.iloc[train_idx], frame.label.iloc[test_idx]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    cv_splits = list(inner.split(X_train, y_train, groups[train_idx]))
    for fit, validation in cv_splits:
        assert not (set(groups[train_idx][fit]) & set(groups[train_idx][validation]))
    results, fitted, cv_rows, warning_log = [], {}, [], []
    print(f"{len(frame)} songs; train={len(train_idx)}, test={len(test_idx)}; artist-disjoint", flush=True)
    # Search and selection use only training folds. The holdout is evaluated after selection.
    with threadpool_limits(limits=1):
        for name, (estimator, grid) in candidates().items():
            tick = time.time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                search = GridSearchCV(estimator, grid, cv=cv_splits, scoring="balanced_accuracy",
                                      refit=True, n_jobs=1, error_score="raise")
                search.fit(X_train, y_train)
            warning_log.extend({"model": name, "warning": str(w.message)} for w in caught)
            fitted[name] = search.best_estimator_
            row = {"model": name, "cv_balanced_accuracy": float(search.best_score_),
                   "cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                   "best_params": search.best_params_, "fit_seconds": round(time.time()-tick, 2)}
            results.append(row)
            for j, params in enumerate(search.cv_results_["params"]):
                cv_rows.append({"model": name, "params": json.dumps(params),
                                "mean_balanced_accuracy": float(search.cv_results_["mean_test_score"][j]),
                                **{f"fold_{k+1}": float(search.cv_results_[f"split{k}_test_score"][j])
                                   for k in range(5)}})
            print(f"{name}: CV balanced accuracy {row['cv_balanced_accuracy']:.3f} ({row['fit_seconds']}s)", flush=True)

        eligible = [r for r in results if r["model"] != "Majority baseline"]
        selected = max(eligible, key=lambda r: r["cv_balanced_accuracy"])["model"]
        for row in results:
            estimator = fitted[row["model"]]
            pred = estimator.predict(X_test)
            scores, score_type = model_scores(estimator, X_test)
            row.update({f"test_{k}": v for k, v in metrics(y_test, pred, scores).items()})
            row["selected"] = row["model"] == selected
            row["confusion_matrix"] = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
    winner = fitted[selected]
    predictions = winner.predict(X_test)
    scores, score_type = model_scores(winner, X_test)
    test_records = frame.iloc[test_idx][["track_id", "artist", "title", "label"]].copy()
    test_records["prediction"] = predictions
    test_records["score"] = scores
    test_records["correct"] = test_records.label == predictions
    test_records.to_csv(RESULTS / "test_predictions.csv", index=False)
    split_manifest = frame[["track_id", "artist", "title", "label"]].copy()
    split_manifest["partition"] = "train"
    split_manifest.loc[test_idx, "partition"] = "test"
    split_manifest.to_csv(RESULTS / "split_manifest.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(RESULTS / "cross_validation.csv", index=False)
    summary = {
        "seed": SEED, "selection_metric": "5-fold training CV balanced accuracy",
        "selected_model": selected, "score_type": score_type,
        "label_definition": "1: source year-end Hot 100 hit; 0: other sampled weekly Hot 100 song",
        "data_audit": data_audit(frame),
        "split": {"train_songs": len(train_idx), "test_songs": len(test_idx),
                  "train_artists": len(set(groups[train_idx])), "test_artists": len(set(groups[test_idx])),
                  "artist_overlap": [], "train_label_counts": y_train.value_counts().sort_index().to_dict(),
                  "test_label_counts": y_test.value_counts().sort_index().to_dict(),
                  "method": "First fold of shuffled 5-fold StratifiedGroupKFold(seed=42), grouped by artist"},
        "models": results,
        "selected_test_ci95_artist_bootstrap": artist_bootstrap(y_test.to_numpy(), predictions, scores, groups[test_idx]),
        "bootstrap_repeats": 2000,
        "warnings": warning_log,
        "versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "librosa", "joblib"]},
        "python": platform.python_version(),
        "elapsed_seconds": round(time.time()-started, 2),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([{k: v for k, v in row.items() if k not in {"best_params", "confusion_matrix"}}
                  for row in results]).to_csv(RESULTS / "model_comparison.csv", index=False)
    # Keep the training-only model so the included test demo remains a genuine holdout.
    joblib.dump({"pipeline": winner, "model_name": selected, "features": FEATURE_COLUMNS,
                 "train_track_ids": frame.track_id.iloc[train_idx].tolist(),
                 "data_sha256": summary["data_audit"]["sha256"], "versions": summary["versions"],
                 "schema": "antonios-518-v1", "score_type": score_type}, MODEL_PATH, compress=3)
    from .plots import make_plots
    make_plots(summary, frame, winner, X_train, test_records, directory=RESULTS / "figures")
    os.rename(staging, destination)
    print(f"Selected {selected} using training CV. Results: {destination}", flush=True)


if __name__ == "__main__":
    main()
