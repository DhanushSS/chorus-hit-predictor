"""HSP-S evidence boundaries and leakage regressions; all test data are synthetic."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest
from chorus_hit.config import ROOT
from chorus_hit.hsp_s import TASK
from chorus_hit.hsp_s.data import audio_columns, identity, label, seal, verify
from chorus_hit.hsp_s.modeling import estimator
from chorus_hit.hit_nonhit.audit import connected_groups
from chorus_hit.hit_nonhit.common import write


def test_only_scalar_low_level_audio_columns_are_predictors():
    schema = pa.schema([('lowlevel.energy.mean', pa.float64()), ('rhythm.bpm', pa.float64()),
                        ('tonal.key_strength', pa.int64()), ('lowlevel.mfcc.mean', pa.list_(pa.float64())),
                        ('metadata.tags.artist_ll', pa.string()), ('highlevel.danceability.probability', pa.float64()),
                        ('peakPos', pa.float64()), ('lastfm_playcount', pa.float64())])
    assert audio_columns(schema) == ['lowlevel.energy.mean', 'rhythm.bpm', 'tonal.key_strength']


@pytest.mark.parametrize('peak,weeks,expected', [(1, 52, 1), (100, 1, 1), (np.nan, np.nan, 0),
                                               (np.nan, 1, None), (50, np.nan, None), (101, 1, None),
                                               (50.5, 1, None), (0, 0, None), (1, -1, None)])
def test_label_policy_does_not_guess_inconsistent_chart_rows(peak, weeks, expected):
    assert label(peak, weeks) == expected


def row(key, artists):
    return {'uuid': key, 'mbid': 'recording-' + key, 'metadata.tags.musicbrainz_artistid_ll': artists,
            'metadata.tags.title_ll': [key], 'metadata.audio_properties.md5_encoded_ll': None}


def test_guests_and_duplicate_features_connect_groups():
    records = [identity(row('a', ['first']), [1., 2.]), identity(row('b', ['first', 'guest']), [2., 3.]),
               identity(row('c', ['guest']), [3., 4.]), identity(row('d', ['other']), [3., 4.]),
               identity(row('e', ['separate']), [5., 6.])]
    groups = connected_groups(records)
    assert len(set(groups[:4])) == 1 and groups[4] != groups[0]
    with pytest.raises(ValueError, match='identity'): identity(row('missing', []), [1., 2.])


def test_imputation_and_scaling_use_training_rows_only():
    cols = ['lowlevel.energy.mean', 'rhythm.bpm']
    X = pd.DataFrame([[1., 2.], [3., 4.], [np.nan, 6.], [5., 8.]], columns=cols)
    c = {'family': 'logistic', 'C': .1, 'representation': 'all_scalar', 'class_weight': 'balanced'}
    model = estimator(c, cols, 1); model.fit(X, [0, 1, 0, 1])
    assert model.named_steps['impute'].statistics_.tolist() == [3., 5.]
    before = model.named_steps['scale'].mean_.copy()
    model.predict(pd.DataFrame([[1e9, 1e9]], columns=cols))
    np.testing.assert_array_equal(before, model.named_steps['scale'].mean_)
    with pytest.raises(ValueError, match='Non-audio'): estimator(c, ['peakPos'], 1)


def test_model_family_grid_stays_separate_from_chorus_config():
    from chorus_hit.hit_nonhit.common import read
    cfg = read(ROOT / 'configs/hsp_s_acousticbrainz.json')
    assert cfg['task_id'] == TASK and len(cfg['candidates']) == 27
    for candidate in cfg['candidates']:
        assert isinstance(estimator(candidate, ['lowlevel.energy.mean', 'rhythm.bpm'], 1), Pipeline)
        assert candidate['representation'] in {'all_scalar', 'summary_scalar', 'pca95'}
    assert read(ROOT / 'configs/hit_nonhit_academic.json')['task_id'] != TASK


def test_manifest_requires_all_files_and_detects_changed_nested_trials(tmp_path):
    for name in ['model.joblib', 'summary.json', 'config.json', 'diagnostics.json']:
        (tmp_path / name).write_text('{}')
    write(tmp_path / 'trials/one.json', {'synthetic': True})
    seal(tmp_path, 'model', {})
    assert verify(tmp_path, 'model')['task_id'] == TASK
    (tmp_path / 'trials/one.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'): verify(tmp_path, 'model')


def test_failed_prediction_consumes_locked_access_and_prevents_retry(tmp_path, monkeypatch):
    from chorus_hit.hsp_s import evaluate as ev
    from chorus_hit.hit_nonhit.common import sha
    split, run = tmp_path / 'split', tmp_path / 'model'
    split.mkdir(); run.mkdir()
    frame = pd.DataFrame({'uuid': ['a', 'b'], 'label': [0, 1], 'group': ['one', 'two'], 'lowlevel.energy.mean': [1., 2.]})
    frame.to_parquet(split / 'locked.parquet', index=False)
    frame.to_parquet(split / 'development.parquet', index=False)
    for name, value in [('config', {'evaluation': {'seed': 1}}), ('split', {'locked': {'ids': ['a', 'b']}}),
                        ('audit', {}), ('columns', ['lowlevel.energy.mean']), ('identity', [])]: write(split / (name + '.json'), value)
    seal(split, 'split', {})
    (run / 'model.joblib').write_bytes(b'test-only fake bundle; never loaded')
    for name, value in [('summary', {}), ('config', {}), ('diagnostics', {'review_flags': []})]: write(run / (name + '.json'), value)
    seal(run, 'model', {'split_manifest_sha256': sha(split / 'manifest.json')})
    class Broken:
        def predict(self, X): raise RuntimeError('test inference failure')
    monkeypatch.setattr(ev, 'load_model', lambda *args, **kwargs: {'columns': ['lowlevel.energy.mean'], 'model': Broken()})
    with pytest.raises(RuntimeError, match='inference failure'): ev.evaluate(split, run)
    assert (split / 'test_access.json').exists()
    with pytest.raises(ValueError, match='already consumed'): ev.evaluate(split, run)


def test_hsp_prediction_rejects_metadata_or_wrong_order(tmp_path, monkeypatch):
    from chorus_hit.hsp_s import evaluate as ev
    monkeypatch.setattr(ev, 'load_model', lambda *args: {'columns': ['lowlevel.energy.mean', 'rhythm.bpm']})
    with pytest.raises(ValueError, match='exact ordered'): ev.predict_frame(tmp_path, pd.DataFrame({'peakPos': [1]}))
    with pytest.raises(ValueError, match='exact ordered'): ev.predict_frame(tmp_path, pd.DataFrame({'rhythm.bpm': [1], 'lowlevel.energy.mean': [2]}))


def test_separate_app_discloses_unavailable_model_and_feature_method(tmp_path, monkeypatch):
    from chorus_hit.hsp_s import ui
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    app = AppTest.from_file(str(ROOT / 'pages/1_Hit_vs_NonHit_Features.py'), default_timeout=30).run()
    assert not app.exception
    assert any('whole-recording Essentia' in x.value for x in app.info)
    assert any('not available yet' in x.value for x in app.info)
