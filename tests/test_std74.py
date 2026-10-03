"""Protect the archive comparison's feature boundary and preprocessing isolation."""
import io
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits
from chorus_hit.config import ROOT, FEATURE_COLUMNS
from chorus_hit.estimators import make_estimator
from chorus_hit.train_v2 import validate_config


def test_std74_matches_independent_archive_pipeline_and_preserves_fit():
    config=json.loads((ROOT/'configs/v4_std74.json').read_text())
    spec=next(s for s in config['candidates'] if s['id']=='v4_lr_std74_quantile_0.1')
    rng=np.random.default_rng(41)
    X=pd.DataFrame(rng.normal(size=(180,518)),columns=FEATURE_COLUMNS)
    y=np.tile([0,1],90)
    selected=[c for c in FEATURE_COLUMNS if '_std_' in c]
    assert len(selected)==74
    reference=make_pipeline(SimpleImputer(strategy='median'),VarianceThreshold(),
        QuantileTransformer(n_quantiles=100,output_distribution='normal',random_state=0),
        LogisticRegression(C=.1,class_weight='balanced',max_iter=3000))
    with threadpool_limits(limits=1):
        actual=make_estimator(spec,FEATURE_COLUMNS).fit(X.iloc[:140],y[:140])
        reference.fit(X[selected].iloc[:140],y[:140])
    assert actual.named_steps['columns'].columns==selected
    assert actual.feature_names_in_.tolist()==FEATURE_COLUMNS
    assert np.array_equal(actual.predict(X.iloc[140:]),reference.predict(X[selected].iloc[140:]))
    assert np.allclose(actual.decision_function(X.iloc[140:]),reference.decision_function(X[selected].iloc[140:]))
    quantiles=actual.named_steps['scale'].quantiles_.copy()
    assert actual.named_steps['scale'].n_quantiles_==100
    assert np.array_equal(quantiles[-1],X[selected].iloc[:140].max().to_numpy())
    actual.predict(X.iloc[140:]*1e9)
    assert np.array_equal(actual.named_steps['scale'].quantiles_,quantiles)
    buf=io.BytesIO();joblib.dump(actual,buf);buf.seek(0)
    assert np.array_equal(joblib.load(buf).predict(X.iloc[140:]),actual.predict(X.iloc[140:]))


def test_v4_retains_all_prior_controls_and_evaluation_membership_rules():
    before=json.loads((ROOT/'configs/v3_robust.json').read_text())
    after=json.loads((ROOT/'configs/v4_std74.json').read_text())
    validate_config(after)
    assert after['candidates'][:55]==before['candidates']
    assert len(after['candidates'])==63
    for key in ['data_sha256','split_sha256','seed','inner_seed','outer_seed',
                'inner_folds','outer_folds','threshold_policy','evaluation_mode']:
        assert after[key]==before[key]
    assert {s['representation']['columns'] for s in after['candidates'][55:]}=={'std_only'}


@pytest.mark.integration
def test_v4_demo_uses_validated_run_and_exploratory_label():
    from streamlit.testing.v1 import AppTest
    from chorus_hit.artifacts import load_run
    run=load_run('v4_std74_001')
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    app.sidebar.selectbox[0].set_value('v4_std74_001').run(timeout=60)
    assert not app.exception
    assert any('v4_std74_001' in c.value for c in app.caption)
    assert any('Exploratory follow-up' in c.value for c in app.caption)
    assert any('63 declared settings' in m.value for m in app.markdown)
    assert any(m.label=='Balanced accuracy' and m.value==f"{run.summary['metrics']['balanced_accuracy']:.1%}" for m in app.metric)
