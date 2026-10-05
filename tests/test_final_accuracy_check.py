import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from chorus_hit.config import ROOT
from chorus_hit.contracts import frozen_split
from chorus_hit.evaluation import grouped_folds
from scripts import final_accuracy_check as final


def test_members_are_fixed_and_drift_is_rejected():
    config = json.loads((ROOT / 'configs/v4_std74.json').read_text())
    specs = final.declared_members(config)
    assert tuple(s['id'] for s in specs) == final.MEMBERS
    assert len(specs) == 8 and final.CONTROL in final.MEMBERS
    changed = copy.deepcopy(config)
    next(s for s in changed['candidates'] if s['id'] == final.CONTROL)['params']['C'] = 3
    with pytest.raises(ValueError, match='settings differ'):
        final.declared_members(changed)
    config['candidates'].append(config['candidates'][0])
    with pytest.raises(ValueError, match='Duplicate'):
        final.declared_members(config)


def test_development_excludes_all_historical_rows():
    from chorus_hit.data import load_data
    config = json.loads((ROOT / 'configs/v4_std74.json').read_text())
    dev, features = final.development_input(config)
    _, dev_ids, historical_ids = frozen_split(load_data())
    assert len(dev) == 597 and len(features) == 518
    assert set(dev.track_id) == set(dev_ids)
    assert not set(dev.track_id) & set(historical_ids)


@pytest.mark.parametrize('bad', [[], [.5], [[np.nan]], [[np.inf]], [[-.1]], [[1.1]]])
def test_invalid_ensemble_scores_rejected(bad):
    with pytest.raises(ValueError, match='probability matrix'):
        final.average_predictions(bad)


def test_equal_weights_and_ties():
    prediction, score = final.average_predictions([[.1, .9], [.4, .8], [.1, .7]])
    np.testing.assert_array_equal(prediction, [0, 1, 0])
    np.testing.assert_allclose(score, [.5, .6, .4])


def test_fitting_only_sees_training_rows_and_control_is_reused(monkeypatch):
    dev = pd.DataFrame({'track_id': [str(i) for i in range(40)],
                        'artist': [f'a{i // 2}' for i in range(40)],
                        'label': [i % 2 for i in range(40)], 'f': np.arange(40)})
    folds = grouped_folds(dev, 5, 44, dev.artist, ['f'])
    calls = []

    class RecordingEstimator:
        classes_ = np.array([0, 1])

        def fit(self, x, y):
            self.seen = set(x.f)
            assert set(y.index) == set(x.index)
            calls.append(self.seen)
            return self

        def predict_proba(self, x):
            assert not self.seen & set(x.f)
            return np.tile([.25, .75], (len(x), 1))

        def predict(self, x):
            assert not self.seen & set(x.f)
            return np.ones(len(x), dtype=int)

    monkeypatch.setattr(final, 'make_estimator', lambda *args: RecordingEstimator())
    specs = [{'id': member} for member in final.MEMBERS]
    rows, records = final.collect_predictions(dev, ['f'], specs, folds)
    assert len(calls) == len(records) == 40
    for i, (fit, val) in enumerate(folds):
        assert all(c == set(dev.f.iloc[fit]) for c in calls[i * 8:(i + 1) * 8])
        assert set(rows.loc[rows.fold == i, 'track_id']) == set(dev.track_id.iloc[val])
        assert not set(dev.artist.iloc[fit]) & set(dev.artist.iloc[val])
    assert rows.track_id.is_unique and len(rows) == len(dev)
    np.testing.assert_array_equal(rows.ensemble_prediction, rows.control_prediction)
    np.testing.assert_allclose(rows.ensemble_score, .75)
    with pytest.raises(ValueError, match='Incomplete'):
        final.collect_predictions(dev, ['f'], specs, folds[:-1])
    with pytest.raises(ValueError, match='overlap or repeat'):
        final.collect_predictions(dev, ['f'], specs, [folds[0], folds[0]])


def test_paired_bootstrap_matches_reference_and_is_deterministic():
    y = np.array([0, 1, 0, 1, 0, 1])
    ensemble = np.array([0, 1, 1, 1, 0, 0])
    control = np.array([1, 1, 0, 0, 1, 0])
    groups = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    result = final.paired_differences(y, ensemble, control, groups, repeats=100)
    assert result == final.paired_differences(y, ensemble, control, groups, repeats=100)
    assert result['metrics']['accuracy']['difference'] == pytest.approx(
        accuracy_score(y, ensemble) - accuracy_score(y, control))
    rng = np.random.default_rng(2026)
    samples = []
    for _ in range(100):
        ix = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(['a', 'b', 'c'], 3)])
        samples.append(accuracy_score(y[ix], ensemble[ix]) - accuracy_score(y[ix], control[ix]))
    np.testing.assert_allclose(result['metrics']['accuracy']['ci95'], np.quantile(samples, [.025, .975]))
    same = final.paired_differences(y, control, control, groups, repeats=100)
    assert all(m == {'difference': 0, 'ci95': [0, 0]} for m in same['metrics'].values())
    with pytest.raises(ValueError, match='positive integer'):
        final.paired_differences(y, ensemble, control, groups, repeats=0)


def test_protocol_written_before_training_and_outputs_never_overwrite(tmp_path, monkeypatch):
    destination = tmp_path / 'run'
    dev = pd.DataFrame({'track_id': ['a', 'b'], 'artist': ['a', 'b'], 'label': [0, 1], 'f': [0, 1]})
    monkeypatch.setattr(final, 'development_input', lambda config: (dev, ['f']))
    monkeypatch.setattr(final, 'grouped_folds', lambda *args: [(np.array([0]), np.array([1]))])
    def stop_before_training(*args):
        protocol = json.loads((destination / 'protocol.json').read_text())['protocol']
        assert protocol['fit_count'] == 40
        assert protocol['weights'] == [.125] * 8
        assert protocol['fresh_test'] is False
        assert 'Stop regardless of outcome' in protocol['stopping_rule']
        raise RuntimeError('intentional stop before any fits')
    monkeypatch.setattr(final, 'collect_predictions', stop_before_training)
    with pytest.raises(RuntimeError, match='intentional stop'):
        final.run(destination)
    assert not (destination / 'result.json').exists()
    original = (destination / 'protocol.json').read_bytes()
    with pytest.raises(FileExistsError):
        final.run(destination)
    assert (destination / 'protocol.json').read_bytes() == original
