import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from chorus_hit.train_v2 import Runner,validate_config


def test_smoke_and_checkpoint_resume(tmp_path):
    f=pd.DataFrame({'track_id':[str(i) for i in range(24)],'artist':[f'g{i//2}' for i in range(24)],
        'label':[i%2 for i in range(24)],'a':np.arange(24),'b':np.arange(24)**2})
    spec={'id':'smoke','family':'logistic','representation':{'kind':'none'},'params':{'C':.1}}
    config={'parallelism':1,'seed':42,'fit_timeout_seconds':30}
    runner=Runner(tmp_path,f,['a','b'],config,time.monotonic()+60)
    task={'key':'smoke_0','stage':'smoke','fold':0,'spec':spec,'fit':list(range(16)),'validation':list(range(16,24))}
    first=runner.execute([task]); assert first[0]['status']=='complete'; assert len(first[0]['predictions'])==8
    digest=(tmp_path/'trials/smoke_0.json').read_bytes(); runner.deadline=time.monotonic()-1
    assert runner.execute([task])==first
    assert (tmp_path/'trials/smoke_0.json').read_bytes()==digest
    task={**task,'key':'timeout_1'}
    with pytest.raises(TimeoutError): runner.execute([task])
    assert not (tmp_path/'trials/timeout_1.json').exists()


def test_budget_and_holdout_guards():
    from chorus_hit.config import ROOT
    c=json.loads((ROOT/'configs/v2_quick.json').read_text()); validate_config(c)
    with pytest.raises(ValueError,match='budget'): validate_config({**c,'max_trials':1})
    with pytest.raises(ValueError,match='development_only'): validate_config({**c,'evaluation_mode':'historical_test'})
