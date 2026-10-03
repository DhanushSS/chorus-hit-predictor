"""Check recorded stability evidence against independent metrics and fold IDs."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score
from chorus_hit.audit import sha256_file
from chorus_hit.config import ROOT, FEATURE_COLUMNS
from chorus_hit.data import normalize_artist


@pytest.mark.integration
def test_stability_evidence_has_disjoint_development_folds_and_independent_scores():
    path = ROOT/'results/stability_001'
    manifest = json.loads((path/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert len(manifest['trials']) == 100
    for name, digest in manifest['files'].items():
        assert sha256_file(path/name) == digest
    predictions = pd.read_csv(path/'predictions.csv')
    metrics = pd.read_csv(path/'repeat_metrics.csv')
    split = pd.read_csv(ROOT/'results/split_manifest.csv')
    allowed = set(split.loc[split.partition=='train','track_id'])
    assert set(predictions.seed) == set(range(10))
    for seed, rows in predictions.groupby('seed'):
        assert rows.track_id.is_unique and set(rows.track_id)==allowed
        assert rows.groupby('artist_group').fold.nunique().max()==1
        for model in ['lr_mi_50','dummy']:
            saved = metrics[(metrics.seed==seed)&(metrics.model==model)].balanced_accuracy.item()
            assert balanced_accuracy_score(rows.label, rows[model]) == pytest.approx(saved,abs=1e-12)
    summary=json.loads((path/'summary.json').read_text())
    assert summary['rows']==597 and summary['artist_groups']==52
    assert summary['repeat_level_confidence_interval'] is None


@pytest.mark.integration
def test_stability_models_reproduce_fold_predictions_and_fit_only_training_rows():
    path=ROOT/'results/stability_001'
    data=pd.read_csv(ROOT/'data/chorus_features.csv').set_index('track_id')
    predictions=pd.read_csv(path/'predictions.csv')
    for fold in json.loads((path/'folds.json').read_text()):
        fit=data.loc[fold['fit_track_ids']]
        valid=data.loc[fold['validation_track_ids']]
        assert not set(fit.index)&set(valid.index)
        assert not set(fit.artist.map(normalize_artist))&set(valid.artist.map(normalize_artist))
        rows=predictions[(predictions.seed==fold['seed'])&(predictions.fold==fold['fold'])].set_index('track_id').loc[valid.index]
        for name in ['lr_mi_50','dummy']:
            model=joblib.load(path/f"{name}_seed{fold['seed']}_fold{fold['fold']}.joblib")
            assert np.array_equal(model.predict(valid[FEATURE_COLUMNS]), rows[name])
            if name=='lr_mi_50':
                expected=model['variance'].transform(model['imputer'].transform(fit[FEATURE_COLUMNS])).mean(axis=0)
                assert np.allclose(model['scale'].mean_,expected)
