"""Current packaged-model, application, and retained training safety checks."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from chorus_hit.config import ROOT
from chorus_hit.hit_nonhit.common import runtime_check
from chorus_hit.hit_nonhit.modeling import cluster_interval
from chorus_hit.hit_nonhit.train import fit
from chorus_hit.hsp_s.evaluate import predict_frame
from chorus_hit.hsp_s.modeling import estimator
from scripts.setup_model import install


def test_packaged_model_reinstalls_idempotently_and_predicts_example():
    run = install()
    before = hashlib.sha256((run / 'manifest.json').read_bytes()).hexdigest()
    assert install() == run
    assert hashlib.sha256((run / 'manifest.json').read_bytes()).hexdigest() == before
    frame = pd.read_csv(ROOT / 'data/example_features.csv')
    result = predict_frame(run, frame)
    assert result['labels'] == ['Billboard chart entry in HSP-S benchmark',
                               'No chart entry in HSP-S benchmark',
                               'Billboard chart entry in HSP-S benchmark',
                               'Billboard chart entry in HSP-S benchmark']
    np.testing.assert_allclose(result['scores'], [1.187263773345118, -.27028194326293875,
                                                1.066465614339061, .5116200719057967], atol=1e-5)


def test_current_app_displays_team_and_saved_test_metrics():
    install()
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == 'Billboard Hit Prediction Using Audio Features'
    assert any('PES2UG24CS154' in c.value and 'PES2UG24CS150' in c.value for c in app.caption)
    assert {m.label: m.value for m in app.metric}['Balanced accuracy'] == '73.17%'
    assert any('MP3' in i.value and 'not supported' in i.value for i in app.info)
    assert app.button[0].disabled


def test_current_app_missing_model_gives_setup_instruction(tmp_path, monkeypatch):
    import chorus_hit.config as config
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert any('scripts/setup_model.py' in w.value for w in app.warning)
    assert not app.metric


def fake_bundle(root, entry='../escaped.txt'):
    (root / 'models').mkdir()
    archive = root / 'models/hsp_s_run_001.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(entry, 'bad')
    (root / 'models/bundle.json').write_text(json.dumps({'files':1, 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(), 'model_manifest_sha256':'unused'}))
    return archive


def test_bundle_tampering_rejected_before_loading(tmp_path):
    archive = fake_bundle(tmp_path)
    archive.write_bytes(archive.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum'): install(tmp_path)
    assert not (tmp_path / 'results/hsp_s/run_001').exists()


def test_bundle_path_traversal_rejected(tmp_path):
    fake_bundle(tmp_path)
    with pytest.raises(ValueError, match='Unsafe archive path'): install(tmp_path)
    assert not (tmp_path / 'results/hsp_s/escaped.txt').exists()


def test_existing_different_model_not_overwritten(tmp_path):
    fake_bundle(tmp_path)
    dest = tmp_path / 'results/hsp_s/run_001'; dest.mkdir(parents=True)
    (dest / 'manifest.json').write_text('preserve me')
    with pytest.raises(ValueError, match='not be overwritten'): install(tmp_path)
    assert (dest / 'manifest.json').read_text() == 'preserve me'


def test_runtime_platform_and_version_fail_closed(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    with pytest.raises(ValueError, match='macOS/Linux'): runtime_check()
    monkeypatch.setattr(sys, 'platform', 'darwin'); monkeypatch.setattr(sys, 'version_info', (3,13,0))
    with pytest.raises(ValueError, match='3.12'): runtime_check()


def test_actual_fit_worker_has_hard_timeout():
    columns = ['lowlevel.energy.mean', 'rhythm.bpm']
    model = estimator({'family':'dummy','representation':'all_scalar'}, columns, 42)
    with pytest.raises(TimeoutError, match='time limit'):
        fit(model, pd.DataFrame([[0.,1.],[1.,2.]],columns=columns), np.array([0,1]), .01)


def test_insufficient_groups_no_invented_interval():
    assert cluster_interval([0,1,0,1],[0,1,1,1],['a','a','b','b'],42)['balanced_accuracy'] is None
