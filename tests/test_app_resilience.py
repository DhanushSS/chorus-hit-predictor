import pytest
from chorus_hit.config import ROOT


def test_v4_is_only_distributed_model_and_demo_predicts():
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from chorus_hit.artifacts import active_run, load_run
    st.cache_resource.clear()
    assert active_run() == 'v4_std74_001'
    assert list((ROOT/'results').rglob('model.joblib')) == [ROOT/'results/v2/v4_std74_001/model.joblib']
    assert not (ROOT/'models/selected_model.joblib').exists()
    app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    assert not app.exception and not app.sidebar.selectbox
    assert any('Chorus Hit Predictor' in c.value for c in app.caption)
    assert len(app.selectbox[0].options) == 154
    app.button[0].click().run(timeout=60)
    assert not app.exception and any(m.label == 'Model prediction' for m in app.metric)
    app.radio[0].set_value('Upload audio').run(timeout=60)
    assert not app.exception
    assert any(b.label == 'Analyze audio' and b.disabled for b in app.button)
    assert load_run().summary['fresh_test_available'] is False


def test_missing_selected_model_stops_predictions(monkeypatch):
    import streamlit as st
    import chorus_hit.artifacts as module
    from streamlit.testing.v1 import AppTest
    def invalid(run): raise ValueError('selected model corrupt')
    monkeypatch.setattr(module,'run_cache_key',invalid);st.cache_resource.clear()
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    assert not app.exception and any('selected model corrupt' in e.value for e in app.error)
    assert not app.button


def test_missing_frozen_split_disables_historical_demo(monkeypatch):
    import streamlit as st
    import chorus_hit.contracts as contracts
    import chorus_hit.artifacts as artifacts
    from streamlit.testing.v1 import AppTest
    # Validate/cache model first: failure here targets only the demo's split read.
    validated = artifacts.load_run()
    monkeypatch.setattr(artifacts, 'load_run', lambda *a, **k: validated)
    def unavailable(*args): raise FileNotFoundError('split absent')
    monkeypatch.setattr(contracts, 'frozen_split', unavailable);st.cache_resource.clear()
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    assert not app.exception and any('Historical song demo unavailable' in e.value for e in app.warning)
    app.radio[0].set_value('Upload audio').run(timeout=60)
    assert not app.exception and any(b.label == 'Analyze audio' for b in app.button)
    st.cache_resource.clear()


def test_unavailable_intervals_do_not_crash_app(monkeypatch):
    import streamlit as st
    import chorus_hit.artifacts as module
    from streamlit.testing.v1 import AppTest
    original=module.load_run
    def no_interval(*args,**kwargs):
        a=original(*args,**kwargs);a.summary['ci95']['balanced_accuracy']=None;return a
    monkeypatch.setattr(module,'load_run',no_interval);st.cache_resource.clear()
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    assert not app.exception
    assert any('Bootstrap interval unavailable' in e.value for e in app.info)
    st.cache_resource.clear()
