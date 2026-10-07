"""C01-C04: invalid input boundaries; all audio here is synthetic."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from chorus_hit.evaluation import classification_metrics, bootstrap_intervals, paired_bootstrap
from chorus_hit.ingestion import FIELDS, validate_record, coverage_report
from chorus_hit.audit import sha256_file


@pytest.mark.parametrize('value', [[.2,1.8],[np.nan,1],[0,np.inf],0,[[0,1]],[],['0','1'],[False,True]])
@pytest.mark.parametrize('position', [0,1])
def test_invalid_binary_inputs(value,position):
    args=[[0,1],[0,1]];args[position]=value
    with pytest.raises(ValueError): classification_metrics(*args)


@pytest.mark.parametrize('scores', [[0],[[0,1]],[np.nan,1],[np.inf,1]])
def test_invalid_scores(scores):
    with pytest.raises(ValueError): classification_metrics([0,1],[0,1],scores)


def test_valid_metrics_and_zero_division():
    y=[0.,0.,1.,1.];p=[0,1,0,1]
    m=classification_metrics(y,p)
    for key,fn in [('accuracy',accuracy_score),('balanced_accuracy',balanced_accuracy_score),('precision',precision_score),('recall',recall_score),('f1',f1_score)]: assert m[key]==fn(y,p)
    assert classification_metrics(y,[0,0,0,0])['precision']==0
    with pytest.raises(ValueError): classification_metrics([0,1],[1])


@pytest.mark.parametrize('fn',[bootstrap_intervals,paired_bootstrap])
@pytest.mark.parametrize('repeats',[0,-1,1.5,True])
def test_bootstrap_repeats(fn,repeats):
    args=([0,1],[0,1],['a','b']) if fn==bootstrap_intervals else ([0,1],[0,1],[1,0],['a','b'])
    with pytest.raises(ValueError,match='positive integer'):fn(*args,repeats=repeats)


@pytest.mark.parametrize('labels,pred,groups', [([],[],[]),([0,0],[0,1],['a','b']),([0,1],[0,1],['a','a']),([0,1],[0,1],['a']),([0,1],[0,1],['a',None]),([0,1],[0,1],[['a','b']]),([0,1],[0],['a','b'])])
def test_bootstrap_invalid_arrays(labels,pred,groups):
    with pytest.raises(ValueError):bootstrap_intervals(labels,pred,groups,repeats=2)
    with pytest.raises(ValueError):paired_bootstrap(labels,pred,pred,groups,repeats=2)


def test_bootstrap_no_valid_draws_and_determinism(monkeypatch):
    y=[0,1,0,1];p=[1,1,0,0];g=['b','a','d','c']
    assert bootstrap_intervals(y,p,g,repeats=30)==bootstrap_intervals(y,p,g,repeats=30)
    assert paired_bootstrap(y,p,y,g,repeats=30)==paired_bootstrap(y,p,y,g,repeats=30)
    class OneGroup:
        def choice(self,values,size,*args,**kwargs):return np.repeat(values[0],size)
    monkeypatch.setattr(np.random,'default_rng',lambda seed:OneGroup())
    assert bootstrap_intervals([0,1],[0,1],['a','b'],repeats=3)['intervals']['balanced_accuracy'] is None
    r=paired_bootstrap([0,1],[0,1],[1,0],['a','b'],repeats=3)
    assert r['ci95'] is None and r['valid_resamples']==0 and r['unavailable_reason']


def audio_record(tmp_path):
    p=tmp_path/'record.wav'
    import soundfile as sf
    sf.write(p,.1*np.sin(np.arange(15*8000)*.1),8000)
    r=dict.fromkeys(FIELDS)
    r.update(recording_id='recording:test',dataset_version='synthetic-v1',title='Fixture',original_artist_credit='Synthetic',
       canonical_artist_ids=['fixture:artist'],identity_evidence_source='https://example.test/identity',
       recording_version='synthetic-original',label=1,label_definition_version='chart-test-v1',label_evidence_status='verified',
       label_source='https://example.test/chart',chart_type='year_end_hot100',chart_market='US',
       observation_start='2020-01-01',observation_end='2020-12-31',audio_path_or_authorized_reference=str(p),
       audio_sha256=sha256_file(p),audio_rights_basis='Synthetic software fixture',extractor_version='shared-librosa-518-v2',
       segment_start_seconds=0,segment_duration_seconds=15)
    return r


@pytest.mark.parametrize('key,value', [('label',None),('label',.2),('label_source',None),('label_source','placeholder'),('identity_evidence_source',None),('canonical_artist_ids',[42]),('canonical_artist_ids',['']),('audio_sha256','bad'),('audio_sha256','0'*64),('audio_path_or_authorized_reference','/absent'),('recording_version',None),('chart_market',None),('chart_type',None),('observation_end','2019-01-01'),('extractor_version','bad'),('segment_start_seconds',-1),('segment_start_seconds',float('nan')),('segment_start_seconds',float('inf')),('segment_duration_seconds',14),('segment_start_seconds',1)])
def test_coverage_equals_strict_validator(tmp_path,key,value):
    r=audio_record(tmp_path);assert coverage_report([r])[0]['eligible'];validate_record(r,require_audio=True)
    r[key]=value
    report=coverage_report([r])[0]; assert not report['eligible'] and report['reasons']
    with pytest.raises((ValueError,FileNotFoundError)):validate_record(r,require_audio=True)


@pytest.mark.parametrize('case',['one_missing','empty_row','all_missing_column','infinity','no_rows'])
def test_missing_features_rejected_before_fit(tmp_path,case):
    from chorus_hit.data import load_data
    from chorus_hit.artifacts import load_run
    f=pd.DataFrame({'track_id':['a','b'],'artist':['x','y'],'title':['a','b'],'label':[0,1],'a':[1.,2.],'b':[3.,4.]})
    if case=='one_missing':f.loc[0,'a']=np.nan
    elif case=='empty_row':f.loc[0,['a','b']]=np.nan
    elif case=='all_missing_column':f['a']=np.nan
    elif case=='infinity':f.loc[0,'a']=np.inf
    else:f=f.iloc[:0]
    path=tmp_path/'data.csv';f.to_csv(path,index=False)
    with pytest.raises(ValueError):load_data(path,features=['a','b'])
    a=load_run('v1_baseline');X=a.data[a.bundle['features']].head(1).copy()
    if case=='no_rows':X=X.iloc[:0]
    else:X.iloc[0,0]=np.nan if case!='infinity' else np.inf
    with pytest.raises(ValueError):a.predict(X)


def test_finite_dataset_fit_save_reload_predict(tmp_path):
    from chorus_hit.data import load_data
    from chorus_hit.estimators import make_estimator
    from chorus_hit.artifacts import RunAssets
    from chorus_hit.evaluation import score_model
    f=pd.DataFrame({'track_id':list('abcd'),'artist':list('abcd'),'title':list('abcd'),'label':[0,1,0,1],'a':[1.,2.,3.,4.],'b':[4.,3.,2.,1.]})
    path=tmp_path/'data.csv';f.to_csv(path,index=False);f=load_data(path,features=['a','b'])
    model=make_estimator({'family':'logistic','representation':{'kind':'none'},'params':{}},['a','b']);model.fit(f[['a','b']],f.label)
    _,semantics=score_model(model,f[['a','b']]);bundle={'pipeline':model,'features':['a','b'],'extractor_version':'fixture','score_semantics':semantics}
    joblib.dump(bundle,tmp_path/'model.joblib');a=RunAssets(tmp_path,{},joblib.load(tmp_path/'model.joblib'),{},f,None)
    assert np.array_equal(a.predict(f[['a','b']])[0],model.predict(f[['a','b']]))
