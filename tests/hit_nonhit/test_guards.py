from pathlib import Path
import json
import shutil
import sys
import importlib
import numpy as np
import pytest
from chorus_hit.config import ROOT
from chorus_hit.hit_nonhit.common import read, sha, verify, runtime_check, digest
from chorus_hit.hit_nonhit.synthetic import feature_package
from chorus_hit.hit_nonhit.freeze_split import freeze
from chorus_hit.hit_nonhit.modeling import deadline, cluster_interval
from chorus_hit.hit_nonhit.artifacts import load
from chorus_hit.hit_nonhit.build_dataset import load_dataset
from chorus_hit.hit_nonhit.schema import candidate


def test_runtime_platform_and_version_fail_closed(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    with pytest.raises(ValueError, match='macOS/Linux'): runtime_check()
    monkeypatch.setattr(sys, 'platform', 'darwin'); monkeypatch.setattr(sys, 'version_info', (3,13,0))
    with pytest.raises(ValueError, match='3.12'): runtime_check()


def test_fit_deadline_is_enforced():
    import time
    with pytest.raises(TimeoutError, match='time limit'):
        with deadline(.02): time.sleep(.2)


def test_actual_fit_worker_has_hard_timeout():
    from chorus_hit.hit_nonhit.train import fit
    from chorus_hit.hit_nonhit.modeling import estimator
    import pandas as pd
    from chorus_hit.config import FEATURE_COLUMNS
    X = pd.DataFrame(np.zeros((2, 518)), columns=FEATURE_COLUMNS)
    model = estimator({'family':'dummy', 'representation':'all518'}, 42)
    with pytest.raises(TimeoutError, match='time limit'):
        fit(model, X, np.array([0,1]), .01)


def test_insufficient_groups_no_invented_ci():
    result = cluster_interval([0,1,0,1],[0,1,1,1],['a','a','b','b'],42)
    assert result['balanced_accuracy'] is None


def test_missing_manifest_and_omitted_file_hash(tmp_path):
    with pytest.raises(FileNotFoundError): verify(tmp_path, 'dataset')
    p = feature_package(tmp_path / 'data')
    m = read(p / 'manifest.json'); del m['files']['features.csv']
    (p / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='omits'): verify(p, 'dataset')


def test_unknown_and_wrong_field_types_rejected():
    from chorus_hit.hit_nonhit.synthetic import inputs
    rows, _, _ = inputs(1)
    rows[0]['canonical_artist_ids'] = 'one_artist'
    with pytest.raises(ValueError, match='artist IDs'): candidate(rows[0])


def test_resume_after_interruption_without_holdout_read(tmp_path, monkeypatch):
    trainer = importlib.import_module('chorus_hit.hit_nonhit.train')
    feature_package(tmp_path / 'data'); freeze(tmp_path / 'data', tmp_path / 'split')
    original = trainer.fit; calls = 0
    def interrupted(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2: raise KeyboardInterrupt('synthetic interruption')
        return original(*a, **kw)
    monkeypatch.setattr(trainer,'fit',interrupted)
    with pytest.raises(KeyboardInterrupt): trainer.train(tmp_path / 'split',tmp_path / 'run')
    assert len(list((tmp_path / 'run/trials').glob('*.json'))) == 1
    assert not (tmp_path / 'run/.training.lock').exists()
    import pandas as pd
    read_csv = pd.read_csv
    def forbid_test(path, *a, **kw):
        assert Path(path).name != 'locked.csv', 'Trainer read sealed test labels'
        return read_csv(path, *a, **kw)
    monkeypatch.setattr(pd,'read_csv',forbid_test)
    result = trainer.train(tmp_path / 'split',tmp_path / 'run',resume=True)
    assert result['evaluation_status'] == 'synthetic_test_only'
    # Rehashed semantic corruption must still be rejected by the task loader.
    import joblib
    p = tmp_path / 'run'; bundle = joblib.load(p / 'model.joblib')
    bundle['task_id'] = 'legacy_year_end'; joblib.dump(bundle,p / 'model.joblib')
    m = read(p / 'manifest.json'); m['files']['model.joblib'] = sha(p / 'model.joblib')
    (p / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='task_id'): load(p,allow_synthetic=True)


def test_synthetic_real_mix_and_extra_metadata_column_blocked(tmp_path):
    p = feature_package(tmp_path / 'data')
    m = read(p / 'manifest.json'); m['synthetic'] = False
    (p / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='provenance mismatch'): load_dataset(p)
    m['synthetic'] = True
    import pandas as pd
    frame = pd.read_csv(p / 'features.csv',float_precision='round_trip'); frame['artist_id'] = 'leak'
    frame.to_csv(p / 'features.csv',index=False,float_format='%.17g')
    m['files']['features.csv'] = sha(p / 'features.csv'); (p / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='feature schema'): load_dataset(p)

@pytest.mark.parametrize('family,representation', [('linear_svm','all518'),('rbf_svm','std74'),('random_forest','all518'),('extra_trees','std74'),('logistic','pca95')])
def test_mandatory_families_and_pca_fit_in_supervised_worker(family, representation):
    import pandas as pd
    from chorus_hit.config import FEATURE_COLUMNS
    from chorus_hit.hit_nonhit.train import fit
    from chorus_hit.hit_nonhit.modeling import estimator
    X = pd.DataFrame(np.random.default_rng(19).normal(size=(20,518)), columns=FEATURE_COLUMNS)
    candidate = {'family':family, 'representation':representation, 'C':.1, 'min_samples_leaf':3, 'class_weight':'balanced'}
    model = estimator(candidate,42)
    seconds, warnings = fit(model,X,np.arange(20)%2,30)
    assert len(model.predict(X.iloc[:3])) == 3 and seconds >= 0
