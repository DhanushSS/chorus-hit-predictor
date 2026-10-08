import json
import shutil
import pandas as pd
import pytest
from zipfile import ZipFile
from chorus_hit.artifacts import load_run,run_location,verify_files,trial_records
from chorus_hit.audit import atomic_json,sha256_file
from chorus_hit.evaluation import classification_metrics,target_status,grouped_folds,score_model
from chorus_hit.estimators import make_estimator


def test_bundle_validation_and_input_order(tmp_path):
    a=load_run('v4_std74_001'); X=a.data[a.bundle['features']].head(2)
    a.predict(X)
    with pytest.raises(ValueError,match='order'): a.predict(X.iloc[:,::-1])
    with pytest.raises(ValueError,match='schema'): a.predict(X.iloc[:,:-1])
    with pytest.raises(ValueError,match='schema'): a.predict(X.assign(extra=0))
    with pytest.raises(ValueError,match='extractor'): a.predict(X,'wrong-extractor')
    copy=tmp_path/'run'; shutil.copytree(a.path,copy)
    (copy/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='altered'): load_run(root=copy)
    shutil.copy(a.path/'summary.json',copy/'summary.json'); (copy/'model.joblib').unlink()
    with pytest.raises(ValueError,match='Missing'): load_run(root=copy)


def test_wrong_run_labels_even_with_updated_hash(tmp_path):
    a=load_run('v4_std74_001'); copy=tmp_path/'run'; shutil.copytree(a.path,copy)
    pred=pd.read_csv(copy/'predictions.csv'); pred.loc[0,'label']=1-pred.loc[0,'label']; pred.to_csv(copy/'predictions.csv',index=False)
    m=json.loads((copy/'manifest.json').read_text()); m['files']['predictions.csv']=sha256_file(copy/'predictions.csv'); atomic_json(copy/'manifest.json',m)
    with pytest.raises(ValueError,match='misaligned'): load_run(root=copy)


def test_strict_targets_and_undefined_metrics():
    m={k:.75 for k in ['accuracy','balanced_accuracy','precision','recall','f1']}
    assert not target_status(m,'nested_development')['target_met_in_this_evaluation']
    assert target_status({k:.75000001 for k in m},'nested_development')['fresh_test_target_met'] is None
    assert classification_metrics([0,0],[0,1])['balanced_accuracy'] is None
    class Unsupported: classes_=[0,1]
    with pytest.raises(ValueError,match='supported'): score_model(Unsupported(),[[1]])


def test_new_feature_schema_and_fold_support():
    f=pd.DataFrame({'track_id':[str(i) for i in range(24)],'label':[i%2 for i in range(24)],'a':list(range(24)),'b':[i*i%11 for i in range(24)],'c':[i%5 for i in range(24)]})
    groups=[i//2 for i in range(24)]
    folds=grouped_folds(f,3,42,groups,['a','b','c'])
    e=make_estimator({'family':'logistic','params':{},'representation':{'kind':'anova','value':2}},['a','b','c'])
    fit,val=folds[0]; e.fit(f.iloc[fit][['a','b','c']],f.iloc[fit].label)
    assert e[:-1].transform(f.iloc[val][['a','b','c']]).shape[1]==2
    assert e.named_steps['scale'].n_samples_seen_==len(fit)
    with pytest.raises(ValueError,match='support'): grouped_folds(f,20,42,groups,['a','b','c'])
    bad=make_estimator({'family':'logistic','representation':{'kind':'anova','value':4}},['a','b','c'])
    with pytest.raises(ValueError,match='exceeds'): bad.fit(f[['a','b','c']],f.label)


def test_compact_trial_archive_checks_original_hashes(tmp_path):
    path=tmp_path/'run';path.mkdir()
    trial=b'{"status":"complete","fold":0}'
    (path/'model.joblib').write_bytes(b'model')
    (path/'summary.json').write_bytes(b'{}')
    from hashlib import sha256
    manifest={'status':'complete','model_file':'model.joblib','summary_file':'summary.json',
              'files':{'model.joblib':sha256(b'model').hexdigest(),
                       'summary.json':sha256(b'{}').hexdigest(),
                       'trials/fit_0.json':sha256(trial).hexdigest()}}
    (path/'manifest.json').write_text(json.dumps(manifest))
    with ZipFile(path/'trials.zip','w') as z:z.writestr('trials/fit_0.json',trial)
    verify_files(path,manifest)
    assert trial_records(path)==[json.loads(trial)]
    (path/'trials').mkdir();(path/'trials/fit_0.json').write_bytes(trial)
    with pytest.raises(ValueError,match='Duplicate'):verify_files(path,manifest)
    (path/'trials/fit_0.json').unlink()
    with ZipFile(path/'trials.zip','w') as z:z.writestr('trials/fit_0.json',b'{}')
    with pytest.raises(ValueError,match='altered'):verify_files(path,manifest)
