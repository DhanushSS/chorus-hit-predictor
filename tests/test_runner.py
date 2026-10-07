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


def sleeping_trial(task):
    import time
    if task.get('raises'):raise RuntimeError('worker failure fixture')
    time.sleep(task['sleep'])
    return {'key':task['key'],'spec':task['spec'],'status':'complete'}


def fake_runner(tmp_path,parallelism=2,limit=.35):
    return Runner(tmp_path,pd.DataFrame(),[],{'parallelism':parallelism,'seed':42,'fit_timeout_seconds':limit},time.monotonic()+30,worker=sleeping_trial)


def fake_task(key,seconds):return {'key':key,'spec':{'id':'fixture'},'stage':'fixture','fold':0,'sleep':seconds}


def test_later_fit_times_out_and_completed_checkpoint_survives(tmp_path):
    import multiprocessing as mp
    before={p.pid for p in mp.active_children()}
    runner=fake_runner(tmp_path)
    with pytest.raises(TimeoutError,match='Per-fit'):
        runner.execute([fake_task('first',.05),fake_task('later',1)])
    assert (tmp_path/'trials/first.json').exists()
    assert not (tmp_path/'trials/later.json').exists()
    assert {p.pid for p in mp.active_children()}==before
    runner.deadline=time.monotonic()-1
    assert runner.execute([fake_task('first',.05)])[0]['status']=='complete'
    with pytest.raises(ValueError,match='Resume'):runner.execute([fake_task('first',.06)])


def test_queue_wait_does_not_consume_per_fit_budget(tmp_path):
    runner=fake_runner(tmp_path,parallelism=1,limit=.3)
    assert len(runner.execute([fake_task(str(i),.12) for i in range(4)]))==4


def test_worker_exception_and_interrupt_cleanup(tmp_path,monkeypatch):
    import multiprocessing as mp
    import chorus_hit.train_v2 as module
    before={p.pid for p in mp.active_children()}
    runner=fake_runner(tmp_path)
    with pytest.raises(RuntimeError,match='worker failure'):runner.execute([{**fake_task('bad',0),'raises':True}])
    assert {p.pid for p in mp.active_children()}==before
    def interrupt(_):raise KeyboardInterrupt()
    monkeypatch.setattr(module,'poll_pause',interrupt)
    with pytest.raises(KeyboardInterrupt):runner.execute([fake_task('interrupted',1)])
    assert {p.pid for p in mp.active_children()}==before


def test_budget_scope_and_bootstrap_config():
    from chorus_hit.config import ROOT
    c=json.loads((ROOT/'configs/v4_std74.json').read_text())
    validate_config({**c,'budget_scope':'per_invocation'})
    for key,value in [('budget_scope','cumulative'),('bootstrap_repeats',0),('bootstrap_repeats',1.5)]:
        with pytest.raises(ValueError):validate_config({**c,key:value})


def test_unsupported_platform_and_process_group_timeout(tmp_path):
    import subprocess,sys,os
    from chorus_hit.supervision import require_supported_platform,supervise
    with pytest.raises(ValueError,match='macOS or Linux'):require_supported_platform('nt')
    require_supported_platform('posix')
    pidfile=tmp_path/'pid'
    code='import os,time;from pathlib import Path;Path('+repr(str(pidfile))+').write_text(str(os.getpid()));time.sleep(10)'
    with pytest.raises(subprocess.TimeoutExpired):supervise([sys.executable,'-c',code],.5,tmp_path)
    pid=int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):os.kill(pid,0)


def test_resume_gets_declared_invocation_budget(monkeypatch):
    import chorus_hit.train_v2 as module
    monkeypatch.setattr(module.time,'monotonic',lambda:100.)
    assert module.invocation_deadline({'timeout_seconds':30},{'elapsed_seconds':900})==130.
    assert module.invocation_deadline({'timeout_seconds':30},{'elapsed_seconds':0})==130.
