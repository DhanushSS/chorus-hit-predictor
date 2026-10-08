from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from streamlit.testing.v1 import AppTest
from chorus_hit.config import ROOT
from chorus_hit.hit_nonhit.audio import extract, contract
from chorus_hit.hit_nonhit.build_dataset import build, load_dataset
from chorus_hit.hit_nonhit.common import sha, read
from chorus_hit.hit_nonhit.synthetic import inputs


def tone(path, seconds=16, sr=22050, stereo=False, silent=False):
    t = np.arange(round(seconds * sr)) / sr
    y = np.zeros(len(t)) if silent else .2 * np.sin(2*np.pi*220*t) + .12 * np.sin(2*np.pi*330*t)
    if stereo: y = np.column_stack([y, y * .7])
    sf.write(path, y, sr, subtype='FLOAT')
    return path


def test_exact_features_stereo_resampling_and_annotation(tmp_path):
    path = tone(tmp_path / 'audio.wav', sr=44100, stereo=True)
    X, segment, meta = extract(path, 1, {'reviewer':'test', 'evidence':'synthetic://tone', 'is_chorus':True, 'start_seconds':1})
    assert X.shape == (1,518) and np.isfinite(X.to_numpy()).all()
    assert len(segment.audio) == 330750 and segment.audio.ndim == 1
    assert meta['selection_method'] == 'manual_reviewed' and meta['sample_rate'] == 22050
    assert meta['original_sample_rate'] == 44100 and meta['original_channels'] == 2
    assert meta['extractor_version'] == 'shared-librosa-518-v2'
    assert meta['segment_start_sample'] == 22050


def test_fallback_not_verified_chorus(tmp_path):
    _, _, meta = extract(tone(tmp_path / 'short.wav'))
    assert meta['selection_method'] == 'fallback_excerpt' and meta['chorus_annotation_status'] == 'unverified'


@pytest.mark.parametrize('case', ['silence','short','long','bounds','nan','infinity','corrupt','codec','annotation'])
def test_invalid_audio_rejected(tmp_path, case):
    p = tmp_path / 'audio.wav'
    if case == 'corrupt': p.write_bytes(b'bad file')
    else: tone(p, seconds=14 if case == 'short' else 481 if case == 'long' else 16, silent=case == 'silence')
    if case in {'nan','infinity'}:
        y, sr = sf.read(p); y[0] = np.nan if case == 'nan' else np.inf; sf.write(p,y,sr,subtype='FLOAT')
    if case == 'codec': q = p.with_suffix('.mp3'); p.rename(q); p = q
    kwargs = {'start': 2} if case == 'bounds' else {'start':0,'annotation':{'reviewer':'t','evidence':'e','is_chorus':True,'start_seconds':4}} if case == 'annotation' else {}
    with pytest.raises((ValueError, RuntimeError)): extract(p, **kwargs)


def test_authorized_import_and_rights_failure_audited(tmp_path):
    rows, archive, cfg = inputs(1)
    audio_root = tmp_path / 'audio'; audio_root.mkdir()
    for r in rows:
        p = tone(audio_root / r['audio_path']); r['audio_sha256'] = sha(p)
    rows[0]['audio_processing_authorized'] = False
    out = tmp_path / 'dataset'; result = build(archive, rows, cfg, audio_root, out)
    assert result['eligible_audio'] == 1 and result['audio_failures'][0]['label'] == 0
    assert result['negative'] == 1  # retain label evidence even when training audio is excluded
    X, eligible, _, _ = load_dataset(out)
    assert len(X) == 1 and eligible[0]['label'] == 1
    with pytest.raises(ValueError, match='already exists'): build(archive, rows, cfg, audio_root, out)


def test_audio_root_traversal_excluded(tmp_path):
    rows, archive, cfg = inputs(1)
    for r in rows: r['audio_path'] = '../outside.wav'
    result = build(archive, rows, cfg, tmp_path / 'audio', tmp_path / 'dataset')
    assert result['eligible_audio'] == 0 and all('escapes' in x['reason'] for x in result['audio_failures'])


def test_app_blocked_view_keeps_legacy_model():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
    assert not app.exception
    assert any('Dataset/model not ready' in x.value for x in app.info)
    assert any('Class 1 means a source year-end' in x.value for x in app.info)
    assert read(ROOT / 'configs/active_run.json')['run_id'] == 'v4_std74_001'
    assert app.title[0].value == 'Chorus Hit Predictor'
