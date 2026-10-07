"""Schema checks and artist-disjoint splits. Metadata never enters a model."""
import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import DATA_PATH, FEATURE_COLUMNS, SEED


def normalize_artist(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def load_data(path=DATA_PATH, *, features=None):
    from .audit import frame_fingerprint, sha256_file
    features = list(FEATURE_COLUMNS if features is None else features)
    path = Path(path).resolve()
    digest = sha256_file(path)
    frame = pd.read_csv(path)
    required = ["track_id", "artist", "title", "label", *features]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)[:8]}")
    if frame.columns.tolist() != required:
        raise ValueError("Dataset columns must match the declared ordered schema exactly")
    if frame.track_id.isna().any():
        raise ValueError("Missing track ID")
    if frame.track_id.duplicated().any():
        raise ValueError("Duplicate track IDs in dataset")
    if frame[["artist", "title", "label"]].isna().any().any():
        raise ValueError("Missing artist, title, or label")
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError("Dataset must contain both binary labels 0 and 1")
    frame[features] = frame[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[features].to_numpy()).all():
        raise ValueError("Audio features must be finite; missing values are not accepted")
    if frame[features].duplicated().any():
        raise ValueError("Duplicate feature vectors: remove or group duplicates before splitting")
    if frame.assign(artist_key=frame.artist.map(normalize_artist)).duplicated(["artist_key", "title"]).any():
        raise ValueError("Duplicate artist/title pairs")
    if sha256_file(path) != digest:
        raise ValueError("Dataset changed while reading")
    frame.attrs.update(source_path=str(path), source_sha256=digest,
                       loaded_fingerprint=frame_fingerprint(frame), features=features)
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


def data_audit(frame, source_path=None):
    from .audit import frame_fingerprint, sha256_file
    source = Path(source_path or frame.attrs.get("source_path", ""))
    if not source.is_file():
        raise ValueError("Audit requires the actual source path from load_data")
    digest = sha256_file(source)
    if digest != frame.attrs.get("source_sha256"):
        raise ValueError("Source hash does not match loaded dataset")
    fingerprint = frame_fingerprint(frame)
    if fingerprint != frame.attrs.get("loaded_fingerprint"):
        raise ValueError("Frame was modified after loading; record a versioned transformation")
    columns = frame.attrs.get("features", FEATURE_COLUMNS)
    features = frame[columns]
    return {
        "rows": len(frame), "features": len(columns),
        "artists": frame.artist.map(normalize_artist).nunique(),
        "label_counts": {str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()},
        "missing_feature_values": int(features.isna().sum().sum()),
        "constant_features": features.columns[features.nunique() <= 1].tolist(),
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "sha256": digest, "source_path": str(source), "processed_fingerprint": fingerprint,
    }
