"""New transforms must fit only training rows and survive model serialization."""
import io
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from chorus_hit.config import ROOT, FEATURE_COLUMNS
from chorus_hit.estimators import make_estimator
from chorus_hit.train_v2 import validate_config


def test_quantiles_are_fitted_to_training_rows_only():
    columns=['mfcc_mean_0','mfcc_std_0','mfcc_kurtosis_0']
    X=pd.DataFrame(np.arange(180).reshape(60,3),columns=columns)
    y=np.tile([0,1],30)
    spec={'family':'logistic','representation':{'kind':'none','scaling':'quantile'}}
    model=make_estimator(spec,columns).fit(X,y)
    before=model.named_steps['scale'].quantiles_.copy()
    extremes=pd.DataFrame([[1e12,-1e12,1e12]],columns=columns)
    assert np.isfinite(model[:-1].transform(extremes)).all()
    assert np.array_equal(model.named_steps['scale'].quantiles_,before)
    assert np.array_equal(before[-1],X.max().to_numpy())


def test_new_candidates_clone_reload_and_keep_raw_schema():
    config=json.loads((ROOT/'configs/v3_robust.json').read_text());validate_config(config)
    previous=json.loads((ROOT/'configs/v2_thorough.json').read_text())
    assert config['candidates'][:35]==previous['candidates']
    rng=np.random.default_rng(10)
    X=pd.DataFrame(rng.normal(size=(120,len(FEATURE_COLUMNS))),columns=FEATURE_COLUMNS)
    y=np.tile([0,1],60)
    with threadpool_limits(limits=1):
        for spec in config['candidates'][35:]:
            model=clone(make_estimator(spec,FEATURE_COLUMNS)).fit(X,y)
            expected=model.predict(X.iloc[:4])
            buffer=io.BytesIO();joblib.dump(model,buffer);buffer.seek(0)
            restored=joblib.load(buffer)
            assert np.array_equal(expected,restored.predict(X.iloc[:4]))
            assert restored.feature_names_in_.tolist()==FEATURE_COLUMNS
            if spec['representation']['columns']=='core_statistics':
                assert len(model.named_steps['columns'].columns)==222
                assert all(c.rsplit('_',2)[-2] in {'mean','median','std'} for c in model.named_steps['columns'].columns)
            if spec['family']=='hist_gradient_boosting':
                assert model.named_steps['model'].do_early_stopping_ is False


def test_invalid_preprocessing_and_random_early_stopping_rejected():
    for representation in [{'kind':'none','scaling':'unknown'}, {'kind':'none','columns':'unknown'}]:
        with pytest.raises(ValueError):make_estimator({'family':'logistic','representation':representation},FEATURE_COLUMNS)
    with pytest.raises(ValueError,match='early stopping'):
        make_estimator({'family':'hist_gradient_boosting','params':{'early_stopping':True},'representation':{'kind':'none'}},FEATURE_COLUMNS)


@pytest.mark.integration
def test_v3_app_identifies_exploratory_study():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    app.sidebar.selectbox[0].set_value('v3_nested_001').run(timeout=60)
    assert not app.exception
    assert any('Exploratory follow-up' in c.value for c in app.caption)
    assert any('55 declared settings' in m.value for m in app.markdown)
    app.button[0].click().run(timeout=60)
    assert not app.exception
