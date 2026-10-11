"""Synthetic guards for the new development-only experiment namespace."""
import json
import numpy as np
import pandas as pd
from chorus_hit.config import ROOT
from chorus_hit.hsp_s.data import development, seal, identity
from chorus_hit.hit_nonhit.audit import connected_groups
from chorus_hit.hit_nonhit.common import environment, digest, write
from chorus_hit.research_v2.run import estimator


def test_development_loader_never_reads_locked_labels(tmp_path,monkeypatch):
    records=[identity({'uuid':str(i),'mbid':str(i),'metadata.tags.musicbrainz_artistid_ll':[str(i)],'metadata.tags.title_ll':[str(i)]},[float(i)]) for i in range(4)]
    groups=connected_groups(records); columns=['lowlevel.energy.mean']; cfg={'synthetic':True}
    pd.DataFrame({'uuid':['0','1'],'label':[0,1],'group':groups[:2],columns[0]:[0.,1.]}).to_parquet(tmp_path/'development.parquet')
    (tmp_path/'locked.parquet').write_bytes(b'intentionally unreadable held-out fixture')
    for name,value in [('columns',columns),('config',cfg),('identity',records),('audit',{}),('split',{'development':{'ids':['0','1']},'locked':{'ids':['2','3']}})]: write(tmp_path/f'{name}.json',value)
    seal(tmp_path,'split',{'config_hash':digest(cfg),'feature_schema_hash':digest(columns),'environment':environment()})
    (tmp_path/'test_access.json').write_text('{"used":true}')
    original=pd.read_parquet; accessed=[]
    def guard(path,*args,**kwargs):
        accessed.append(path.name)
        assert path.name=='development.parquet'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',guard)
    frame,_,_,_=development(tmp_path)
    assert len(frame)==2 and accessed==['development.parquet']
    assert (tmp_path/'test_access.json').read_text()=='{"used":true}'


def test_research_preprocessing_and_selection_fit_only_training():
    model=estimator({'family':'svm','scaler':'standard','select_percentile':75},42)
    X=pd.DataFrame([[1.,0.,3.,4.],[3.,1.,5.,2.],[np.nan,2.,7.,8.],[5.,3.,9.,6.]])
    model.fit(X,[0,1,0,1]); original=model.named_steps['scale'].mean_.copy()
    selected=model.named_steps['select'].get_support().copy()
    assert model.named_steps['impute'].statistics_[0]==3
    model.predict(pd.DataFrame([[1e12,1e12,1e12,1e12]]))
    np.testing.assert_array_equal(model.named_steps['scale'].mean_,original)
    np.testing.assert_array_equal(model.named_steps['select'].get_support(),selected)


def test_registered_search_has_finite_budget_and_fixed_seed():
    c=json.loads((ROOT/'configs/research_v2.json').read_text())
    assert c['seed']==20261008 and c['outer_folds']==5 and c['inner_folds']==3
    assert len(c['candidates'])==11 and len({v['id'] for v in c['candidates']})==11
    assert c['per_fit_seconds']==90 and c['max_trial_seconds']==7200
