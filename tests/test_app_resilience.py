import pytest
from chorus_hit.config import ROOT


@pytest.mark.parametrize('selected',['v1_baseline','v4_std74_001'])
def test_missing_optional_comparison_keeps_selected_prediction(monkeypatch,selected):
    import streamlit as st
    import chorus_hit.artifacts as module
    from streamlit.testing.v1 import AppTest
    original=module.run_cache_key
    def unavailable(run):
        if run=='v2_nested_001':raise FileNotFoundError('optional fixture absent')
        return original(run)
    monkeypatch.setattr(module,'run_cache_key',unavailable);st.cache_resource.clear()
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    app.sidebar.selectbox[0].set_value(selected).run(timeout=60)
    assert not app.exception
    assert any('Optional V2 comparison unavailable' in v.value for v in app.info)
    app.button[0].click().run(timeout=60)
    assert not app.exception and any(m.label=='Model prediction' for m in app.metric)


def test_missing_selected_model_stops_predictions(monkeypatch):
    import streamlit as st
    import chorus_hit.artifacts as module
    from streamlit.testing.v1 import AppTest
    def invalid(run):raise ValueError('selected model corrupt')
    monkeypatch.setattr(module,'run_cache_key',invalid);st.cache_resource.clear()
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    assert not app.exception and any('selected model corrupt' in e.value for e in app.error)
    assert not app.button


def test_missing_frozen_baseline_only_disables_historical_demo(monkeypatch):
    import streamlit as st
    import chorus_hit.artifacts as module
    from streamlit.testing.v1 import AppTest
    original=module.run_cache_key
    monkeypatch.setattr(module,'active_run',lambda:'v4_std74_001')
    def unavailable(run):
        if run=='v1_baseline':raise FileNotFoundError('baseline absent')
        return original(run)
    monkeypatch.setattr(module,'run_cache_key',unavailable);st.cache_resource.clear()
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    assert not app.exception and any('Historical song demo unavailable' in e.value for e in app.warning)
    assert any(m.label=='Balanced accuracy' for m in app.metric)


def test_unavailable_intervals_do_not_crash_app(monkeypatch):
    import streamlit as st
    import chorus_hit.artifacts as module
    import chorus_hit.evaluate_run as history
    from streamlit.testing.v1 import AppTest
    original=module.load_run;historical=history.load_historical
    def no_interval(*args,**kwargs):
        a=original(*args,**kwargs);a.summary['ci95']['balanced_accuracy']=None;return a
    def no_paired(*args,**kwargs):
        s=historical(*args,**kwargs);s['paired_vs_v1']['ci95']=None;return s
    monkeypatch.setattr(module,'load_run',no_interval);monkeypatch.setattr(history,'load_historical',no_paired);st.cache_resource.clear()
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    assert not app.exception
    assert any('Bootstrap interval unavailable' in e.value for e in app.info)
    assert any('Paired interval unavailable' in e.value for e in app.info)
    st.cache_resource.clear()
