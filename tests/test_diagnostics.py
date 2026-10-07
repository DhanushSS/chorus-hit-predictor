import json
import subprocess
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.dummy import DummyClassifier
from chorus_hit.config import ROOT
from chorus_hit.diagnostics import new_output,bounded_command


def test_imports_have_no_training_writes_or_warning_filter_effects(tmp_path):
    code='''
import warnings
import sklearn,joblib,numpy,pandas
from pathlib import Path
import chorus_hit.artifacts as a
from sklearn.base import BaseEstimator
from unittest.mock import patch
before=list(warnings.filters)
def fail(*args,**kwargs):raise AssertionError('import side effect')
with patch.object(a,'load_run',fail), patch.object(Path,'mkdir',fail), patch.object(Path,'write_text',fail), patch.object(joblib,'dump',fail):
 import scripts.benchmark_models
 import scripts.leakage_demo
 import scripts.final_accuracy_check
assert warnings.filters==before
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_outputs_refuse_overwrite_and_record_timeout(tmp_path):
    p=tmp_path/'existing';new_output(p);(p/'keep').write_text('original')
    with pytest.raises(FileExistsError):new_output(p)
    with pytest.raises(FileExistsError):bounded_command([sys.executable,'-c','pass'],p,1)
    assert (p/'keep').read_text()=='original'
    timeout=tmp_path/'timeout'
    with pytest.raises(subprocess.TimeoutExpired):bounded_command([sys.executable,'-c','import time;time.sleep(10)'],timeout,.2)
    assert json.loads((timeout/'execution.json').read_text())['status']=='incomplete'


def test_benchmark_rejects_changed_split(tmp_path):
    from scripts.benchmark_models import benchmark
    c=json.loads((ROOT/'configs/v4_std74.json').read_text())
    split=tmp_path/'split.csv';split.write_bytes((ROOT/c['split_manifest']).read_bytes()+b'\n')
    c['split_manifest']=str(split);config=tmp_path/'config.json';config.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='split manifest hash'):benchmark(tmp_path/'benchmark',config,prediction_repeats=1,bootstrap_repeats=1)


class WarnEstimator(ClassifierMixin,BaseEstimator):
    def fit(self,X,y):
        warnings.warn('diagnostic fixture',UserWarning);self.classes_=np.array([0,1]);return self
    def predict(self,X):return np.zeros(len(X),int)


class FailedEstimator(WarnEstimator):
    def fit(self,X,y):raise RuntimeError('deliberate fixture failure')


def fixture_frame():
    return pd.DataFrame({'track_id':[str(i) for i in range(40)],'artist':[f'a{i//2}' for i in range(40)],'label':[i%2 for i in range(40)],'f':np.arange(40)})


def test_small_diagnostic_records_protocol_warning_and_failure(tmp_path):
    from scripts.leakage_demo import compare,validate_settings,models
    r=compare(fixture_frame(),['f'],[42],{'fixture':WarnEstimator()},tmp_path)
    assert len(r)==2 and all(x['status']=='complete' and x['warnings'] for x in r)
    assert {x['protocol'] for x in r}=={'random_song_with_artist_overlap','normalized_artist_name_grouped'}
    assert models(2)['Random forest'].n_jobs==2
    with pytest.raises(ValueError):validate_settings(11,42,1)
    with pytest.raises(ValueError):validate_settings(2,42,5)
    with pytest.raises(RuntimeError,match='deliberate'):compare(fixture_frame(),['f'],[42],{'fixture':FailedEstimator()},tmp_path)
    assert json.loads((tmp_path/'records.json').read_text())[-1]['status']=='failed'
