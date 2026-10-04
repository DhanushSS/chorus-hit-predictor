"""Ceiling check: how high can balanced accuracy go with these 518 features?

Compares a leaky random split with the artist-disjoint split used in the main
experiment, over 5 repeats. Run: python scripts/leakage_demo.py
Writes results/leakage_demo.csv (all numbers are cross-validated, no test-set tuning).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from chorus_hit.config import FEATURE_COLUMNS, RESULTS
from chorus_hit.data import load_data, normalize_artist

warnings.filterwarnings("ignore")
REPEATS = 5


def models():
    return {
        "Logistic regression": make_pipeline(StandardScaler(), PCA(0.95), LogisticRegression(C=0.1, max_iter=3000)),
        "RBF SVM": make_pipeline(StandardScaler(), SVC(C=1)),
        "Random forest": RandomForestClassifier(500, random_state=0, n_jobs=-1),
        "Extra trees": ExtraTreesClassifier(500, random_state=0, n_jobs=-1),
        "Hist gradient boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=0),
    }


def main():
    frame = load_data()
    X, y = frame[FEATURE_COLUMNS].to_numpy(), frame.label.to_numpy()
    groups = frame.artist.map(normalize_artist).to_numpy()
    rows = []
    for name, model in models().items():
        leaky, honest = [], []
        for seed in range(REPEATS):
            p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
            leaky.append(balanced_accuracy_score(y, p))
            p = cross_val_predict(model, X, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed), groups=groups)
            honest.append(balanced_accuracy_score(y, p))
        rows.append({"model": name, "random_split_ba": np.mean(leaky), "random_split_sd": np.std(leaky),
                     "artist_disjoint_ba": np.mean(honest), "artist_disjoint_sd": np.std(honest)})
        print(f"{name:24s} random={np.mean(leaky):.3f}  artist-disjoint={np.mean(honest):.3f}", flush=True)
    out = pd.DataFrame(rows).round(4)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "leakage_demo.csv", index=False)
    print(f"Best artist-disjoint BA: {out.artist_disjoint_ba.max():.3f}")


if __name__ == "__main__":
    main()
