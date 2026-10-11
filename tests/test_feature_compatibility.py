import builtins
import hashlib
import json
import numpy as np
import pytest
from chorus_hit.config import ROOT
from chorus_hit.hsp_s.evaluate import load_model
from chorus_hit.audio_upload.feature_contract import map_features, audio_prediction
from chorus_hit.audio_upload.extract_essentia import extract
from scripts.check_audio_feature_parity import compare


def test_exact_436_mapping_order_and_no_invented_keys():
    columns=load_model(ROOT/'results/hsp_s/run_001')['columns']; pool={key:i for i,key in enumerate(reversed(columns))}
    assert map_features(pool,columns)==list(reversed(range(436)))
    del pool[columns[0]]
    with pytest.raises(ValueError,match='435/436'): map_features(pool,columns)

@pytest.mark.parametrize('bad',[True,'0',[1,2],np.nan,np.inf])
def test_invalid_extractor_scalar(bad):
    with pytest.raises(ValueError): map_features({'a':bad},['a'])

def test_numeric_parity_and_fail_closed_even_after_pair_pass():
    ok=compare({'a':1.},{'a':1.},['a']); bad=compare({'a':1.},{'a':2.},['a'])
    assert ok['numeric_pair_pass'] and not ok['audio_inference_enabled']
    assert not bad['numeric_pair_pass']
    with pytest.raises(ValueError,match='not validated'): audio_prediction(proof=ok,validated=True)

def test_optional_extractor_missing_preserves_graceful_fallback(monkeypatch):
    original=builtins.__import__
    def missing(name,*args,**kwargs):
        if name.startswith('essentia'): raise ImportError('absent in core environment')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',missing)
    with pytest.raises(ValueError,match='CSV and MP3 preview remain usable'): extract('unused.mp3')


def test_frontend_modules_do_not_change_frozen_model_identity():
    from chorus_hit.hsp_s.train import implementation
    assert all('/audio_upload/' not in k and '/ui_components/' not in k for k in implementation())
    manifest=json.loads((ROOT/'results/hsp_s/run_001/manifest.json').read_text())
    assert implementation()==manifest['implementation']
    assert hashlib.sha256((ROOT/'models/hsp_s_run_001.zip').read_bytes()).hexdigest()=='4c335301d937ecacbdc53445d435443f5aa19badbb84ad6bb386c51a81b9f7d3'
    assert load_model(ROOT/'results/hsp_s/run_001')['columns']
