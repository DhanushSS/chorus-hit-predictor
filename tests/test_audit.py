"""Integrity regressions, independent of empirical classifier performance."""
import json
from pathlib import Path
import pandas as pd
import pytest
from chorus_hit.audit import load_dataset,sha256_file
from chorus_hit.data import load_data,data_audit
from chorus_hit.config import DATA_PATH
from scripts.prepare_data import prepare


def test_actual_source_hash_and_mutated_frame(tmp_path):
    original=pd.read_csv(DATA_PATH)
    p=tmp_path/'one.csv'; q=tmp_path/'two.csv'
    original.to_csv(p,index=False); original.iloc[::-1].to_csv(q,index=False)
    a,b=load_data(p),load_data(q)
    assert data_audit(a)['sha256']==sha256_file(p)
    assert data_audit(b)['sha256']==sha256_file(q)
    assert data_audit(a)['sha256']!=data_audit(b)['sha256']
    with pytest.raises(ValueError,match='hash'): data_audit(a,q)
    a.loc[0,'label']=1-a.loc[0,'label']
    with pytest.raises(ValueError,match='modified'): data_audit(a)
    ctx=load_dataset(q); q.write_text(q.read_text()+'\n')
    with pytest.raises(ValueError,match='bytes changed'): ctx.verify()
    with pytest.raises(ValueError,match='SHA-256'): load_dataset(q,expected_sha256='0'*64)


def test_alternate_cannot_inherit_provenance_and_ids_are_stable():
    f=pd.read_csv(DATA_PATH).head(4).drop(columns='track_id').rename(columns={'artist':'Artist','title':'Title','label':'Label'})
    f.insert(0,'choruspath','unknown'); f.insert(0,'Path','unknown')
    raw=f.to_csv(index=False).encode()
    with pytest.raises(ValueError,match='pinned'): prepare(raw)
    with pytest.raises(ValueError,match='requires'): prepare(raw,{})
    prov={'dataset_version':'test-v2','source_name':'fixture','source_url':'fixture://test',
          'label_definition':'synthetic import test only','recording_version':'fixture',
          'license':'fixture','evidence_status':'software-test'}
    a,ap=prepare(raw,prov); b,bp=prepare(f.iloc[::-1].to_csv(index=False).encode(),prov)
    assert a.set_index('title').track_id.to_dict()==b.set_index('title').track_id.to_dict()
    assert 'source_commit' not in ap
    assert ap['source_sha256']!=bp['source_sha256']
