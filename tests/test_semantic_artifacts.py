"""C05: rehashed corruption must still fail semantic verification."""
import json
import shutil
import pandas as pd
import pytest
from chorus_hit.artifacts import load_run, run_location
from chorus_hit.audit import atomic_json, sha256_file
from chorus_hit.config import ROOT


@pytest.mark.parametrize('mutation',['label','group','fold','rows','winner','support','confusion','n','status','count','outer_winner','fit_membership','required'])
def test_nested_semantic_corruption(tmp_path,mutation):
    p=tmp_path/'run';shutil.copytree(run_location('v4_std74_001'),p)
    name='predictions.csv'
    if mutation in {'label','group','fold','rows','winner'}:
        f=pd.read_csv(p/name)
        if mutation=='rows':f=f.iloc[1:]
        else:
            col,value={'label':('label',1-f.loc[0,'label']),'group':('artist_group','invented'),'fold':('fold',4),'winner':('candidate_id','invented')}[mutation]
            f.loc[0,col]=value
        f.to_csv(p/name,index=False)
    else:
        name='outer_fold_metrics.json' if mutation=='outer_winner' else 'outer_folds.json' if mutation=='fit_membership' else 'summary.json'
        s=json.loads((p/name).read_text())
        if mutation=='support':s['metrics']['support']['0']+=1
        if mutation=='confusion':s['metrics']['confusion_matrix'][0][0]+=1
        if mutation=='n':s['metrics']['n']+=1
        if mutation=='status':s['evaluation_status']='fresh_test'
        if mutation=='count':s['counts']['evaluation_songs']+=1
        if mutation=='outer_winner':s[0]['candidate_id']='invented'
        if mutation=='fit_membership':s[0]['fit_track_ids'].append(s[0]['validation_track_ids'][0])
        if mutation=='required':del s['metrics']
        atomic_json(p/name,s)
    m=json.loads((p/'manifest.json').read_text());m['files'][name]=sha256_file(p/name);atomic_json(p/'manifest.json',m)
    with pytest.raises(ValueError):load_run(root=p)


@pytest.mark.parametrize('run',['v4_std74_001'])
def test_each_saved_run_predicts_and_preserves_active(run):
    active=(ROOT/'configs/active_run.json').read_bytes()
    a=load_run(run);p,s,_=a.predict(a.data[a.bundle['features']].head(2))
    assert len(p)==len(s)==2
    assert (ROOT/'configs/active_run.json').read_bytes()==active
