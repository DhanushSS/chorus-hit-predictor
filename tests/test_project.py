import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from chorus_hit.config import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, RESULTS, ROOT, SAMPLE_RATE
from chorus_hit.data import load_data, split_data
from chorus_hit.train import metrics, model_scores


def test_dataset_provenance_and_no_metadata_leakage():
    data = load_data()
    provenance = json.loads((ROOT / "data" / "provenance.json").read_text())
    assert hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() == provenance["prepared_sha256"]
    assert data.shape == (751, 522)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 518
    assert not {"artist", "title", "label", "track_id", "Path", "choruspath"} & set(FEATURE_COLUMNS)


def test_no_artist_overlap_in_holdout_or_cv():
    data = load_data()
    train, test, groups = split_data(data)
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(data)
    for fit, validation in StratifiedGroupKFold(5, shuffle=True, random_state=43).split(
            data.iloc[train][FEATURE_COLUMNS], data.label.iloc[train], groups[train]):
        assert set(groups[train][fit]).isdisjoint(groups[train][validation])


def test_saved_model_and_metrics_are_from_the_holdout():
    data = load_data()
    train, test, _ = split_data(data)
    bundle = joblib.load(MODEL_PATH)
    report = json.loads((RESULTS / "metrics.json").read_text())
    assert set(bundle["train_track_ids"]) == set(data.track_id.iloc[train])
    assert bundle["pipeline"].named_steps["scale"].n_samples_seen_ == len(train)
    assert bundle["pipeline"].feature_names_in_.tolist() == FEATURE_COLUMNS
    X = data.iloc[test][FEATURE_COLUMNS]
    pred = bundle["pipeline"].predict(X)
    score, _ = model_scores(bundle["pipeline"], X)
    computed = metrics(data.label.iloc[test], pred, score)
    selected = next(r for r in report["models"] if r["selected"])
    for key, value in computed.items():
        assert selected[f"test_{key}"] == pytest.approx(value)
    eligible = [r for r in report["models"] if r["model"] != "Majority baseline"]
    assert selected["cv_balanced_accuracy"] == max(r["cv_balanced_accuracy"] for r in eligible)


@pytest.fixture(scope="module")
def musical_signal():
    # Synthetic notes exercise signal processing only. They are never training data.
    sr = SAMPLE_RATE
    t = np.arange(15*sr)/sr
    y = .15*np.sin(2*np.pi*220*t) + .08*np.sin(2*np.pi*330*t) + .04*np.sin(2*np.pi*440*t)
    return (y*(.6+.4*np.sin(2*np.pi*2*t)**2)).astype(np.float32)


def test_audio_features_match_schema_and_allow_inference(musical_signal):
    from chorus_hit.audio import extract_features
    X = extract_features(musical_signal)
    assert X.columns.tolist() == FEATURE_COLUMNS
    assert X.shape == (1, 518)
    assert np.isfinite(X.to_numpy()).all()
    bundle = joblib.load(MODEL_PATH)
    assert int(bundle["pipeline"].predict(X)[0]) in (0, 1)


def test_segment_boundaries_and_invalid_audio(musical_signal, tmp_path):
    from chorus_hit.audio import extract_features, load_audio, select_segment
    import soundfile as sf
    with pytest.raises(ValueError, match="15"):
        select_segment(musical_signal[:SAMPLE_RATE])
    with pytest.raises(ValueError, match="outside"):
        select_segment(musical_signal, start_seconds=1)
    with pytest.raises(ValueError, match="silent"):
        extract_features(np.zeros(15*SAMPLE_RATE))
    longer = np.concatenate([musical_signal, musical_signal, musical_signal])
    segment = select_segment(longer)
    assert len(segment.audio) == 15*SAMPLE_RATE
    assert 0 <= segment.start_seconds <= 30
    path = tmp_path / "test.wav"
    sf.write(path, musical_signal, SAMPLE_RATE)
    loaded, sr = load_audio(path)
    assert sr == SAMPLE_RATE and len(loaded) == len(musical_signal)


def test_demo_runs_and_predicts():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    assert not app.exception
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert app.success or app.warning
