"""Schema checks and artist-disjoint splits. Metadata never enters a model."""
import hashlib
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import DATA_PATH, FEATURE_COLUMNS, SEED


def normalize_artist(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def load_data(path=DATA_PATH):
    frame = pd.read_csv(path)
    required = ["track_id", "artist", "title", "label", *FEATURE_COLUMNS]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)[:8]}")
    if frame.track_id.duplicated().any():
        raise ValueError("Duplicate track IDs in dataset")
    if frame[["artist", "title", "label"]].isna().any().any():
        raise ValueError("Missing artist, title, or label")
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError("Dataset must contain both binary labels 0 and 1")
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if np.isinf(frame[FEATURE_COLUMNS].to_numpy()).any():
        raise ValueError("Infinite audio feature values")
    if frame[FEATURE_COLUMNS].duplicated().any():
        raise ValueError("Duplicate feature vectors: remove or group duplicates before splitting")
    if frame.assign(artist_key=frame.artist.map(normalize_artist)).duplicated(["artist_key", "title"]).any():
        raise ValueError("Duplicate artist/title pairs")
    return frame


def split_data(frame):
    groups = frame.artist.map(normalize_artist).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(outer.split(frame[FEATURE_COLUMNS], frame.label, groups))
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("An artist appears in both training and test data")
    for indices in (train, test):
        if len(np.unique(frame.label.iloc[indices])) != 2:
            raise ValueError("A partition has only one label")
    return train, test, groups


def data_audit(frame):
    features = frame[FEATURE_COLUMNS]
    return {
        "rows": len(frame), "features": len(FEATURE_COLUMNS),
        "artists": frame.artist.map(normalize_artist).nunique(),
        "label_counts": {str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()},
        "missing_feature_values": int(features.isna().sum().sum()),
        "constant_features": features.columns[features.nunique() <= 1].tolist(),
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    }
