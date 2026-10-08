"""Bounded, resumable development experiments; historical labels never enter tuning."""
import argparse
import os
import signal
import sys
from datetime import datetime,timezone
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time
import traceback
import warnings
import queue
from time import sleep as poll_pause
from .supervision import require_supported_platform, supervise

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .audit import load_dataset,atomic_json,sha256_file,json_hash,LEGACY_DATASET,TASK_VERSION,GROUP_VERSION
from .artifacts import RUNS,EXTRACTOR_VERSION,finalize_run,run_location
from .config import ROOT,LABELS
from .data import normalize_artist
from .estimators import make_estimator
from .evaluation import classification_metrics,score_model,grouped_folds,target_status,bootstrap_intervals,positive_integer

_WORKER=None
_EVENTS=None


def worker_init(frame,features,seed,events=None):
    global _WORKER, _EVENTS
    _WORKER=(frame,features,seed)
    _EVENTS=events


def timed_trial(task, worker):
    started=time.monotonic()
    _EVENTS.put((task['key'], started))
    result=worker(task)
    return result, time.monotonic()-started


def fit_trial(task):
    frame,features,seed=_WORKER; t=time.monotonic()
    result={k:task[k] for k in ['key','stage','fold','spec']}
    train=frame.iloc[task['fit']]; val=frame.iloc[task['validation']]
    result.update(fit_rows=len(train),validation_rows=len(val),fit_groups=train.artist.map(normalize_artist).nunique(),
                  validation_groups=val.artist.map(normalize_artist).nunique())
    caught=[]
    try:
        with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model=make_estimator(task['spec'],features,seed)
            model.fit(train[features],train.label)
            p=model.predict(val[features]); score,semantics=score_model(model,val[features])
            train_pred=model.predict(train[features])
            result.update(status='complete',train_metrics=classification_metrics(train.label,train_pred),
                metrics=classification_metrics(val.label,p,score),dimensions=int(model[:-1].transform(train[features].iloc[:1]).shape[1]),
                predictions=[{'track_id':r.track_id,'artist_group':normalize_artist(r.artist),'label':int(r.label),
                              'prediction':int(p[i]),'score':float(score[i]),'score_kind':semantics['kind']}
                             for i,r in enumerate(val.itertuples())])
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    result['warnings']=[f'{type(w.message).__name__}: {w.message}' for w in caught]
    result['seconds']=time.monotonic()-t
    return result


class Runner:
    def __init__(self,staging,frame,features,config,deadline,worker=fit_trial):
        self.worker=worker
        self.staging=staging; self.frame=frame; self.features=features; self.config=config; self.deadline=deadline
        (staging/'trials').mkdir(exist_ok=True)

    def execute(self,tasks):
        completed=[]; pending=[]
        for task in tasks:
            path=self.staging/'trials'/f"{task['key']}.json"
            if path.exists():
                r=json.loads(path.read_text())
                if r['spec']!=task['spec'] or r.get('task_sha256')!=json_hash(task): raise ValueError('Resume trial/config mismatch')
                completed.append(r)
            else: pending.append(task)
        if not pending: return completed
        context=mp.get_context('spawn'); events=context.Queue(); starts={}
        if time.monotonic()>=self.deadline: raise TimeoutError('Declared invocation wall-time budget exhausted')
        pool=context.Pool(self.config['parallelism'],initializer=worker_init,
            initargs=(self.frame,self.features,self.config['seed'],events))
        jobs={task['key']:(task,pool.apply_async(timed_trial,(task,self.worker))) for task in pending}
        try:
            while jobs:
                while True:
                    try:
                        key,started=events.get_nowait(); starts[key]=started
                    except queue.Empty: break
                overdue=[]
                for key,(task,job) in list(jobs.items()):
                    if job.ready():
                        result,seconds=job.get()
                        if seconds>self.config['fit_timeout_seconds']:
                            overdue.append(key); continue
                        result['execution_seconds']=seconds
                        result['task_sha256']=json_hash(task)
                        atomic_json(self.staging/'trials'/f'{key}.json',result)
                        completed.append(result); del jobs[key]
                    elif key in starts and time.monotonic()-starts[key]>self.config['fit_timeout_seconds']:
                        overdue.append(key)
                if overdue: raise TimeoutError(f'Per-fit execution budget exhausted: {overdue}')
                if time.monotonic()>=self.deadline: raise TimeoutError('Declared invocation wall-time budget exhausted')
                if jobs: poll_pause(.01)
            pool.close(); pool.join()
        except BaseException:
            pool.terminate(); pool.join(); raise
        finally:
            events.close(); events.join_thread()
        return completed

    def search(self,stage,indices,fold_seed):
        subset=self.frame.iloc[indices].reset_index(drop=True)
        groups=subset.artist.map(normalize_artist).to_numpy()
        folds=grouped_folds(subset,self.config['inner_folds'],fold_seed,groups,self.features)
        fold_path=self.staging/f'{stage}_folds.json'
        assignment=[{'fold':i,'fit_track_ids':subset.track_id.iloc[fit].tolist(),
                    'validation_track_ids':subset.track_id.iloc[val].tolist()} for i,(fit,val) in enumerate(folds)]
        if fold_path.exists() and json.loads(fold_path.read_text())!=assignment: raise ValueError('Resume fold mismatch')
        atomic_json(fold_path,assignment)
        tasks=[]
        for spec in self.config['candidates']:
            for i,(fit,val) in enumerate(folds):
                tasks.append({'key':f'{stage}_{spec["id"]}_{i}','stage':stage,'fold':i,'spec':spec,
                    'fit':np.asarray(indices)[fit].tolist(),'validation':np.asarray(indices)[val].tolist()})
        trials=self.execute(tasks); ranking=[]
        for spec in self.config['candidates']:
            rows=[r for r in trials if r['spec']['id']==spec['id']]
            success=[r for r in rows if r['status']=='complete']
            eligible=len(success)==len(folds)
            ranking.append({'stage':stage,'candidate_id':spec['id'],'family':spec['family'],
                'representation':json.dumps(spec['representation'],sort_keys=True),'eligible':eligible,
                'mean_balanced_accuracy':float(np.mean([r['metrics']['balanced_accuracy'] for r in success])) if eligible else None,
                'fold_std':float(np.std([r['metrics']['balanced_accuracy'] for r in success])) if eligible else None,
                'mean_train_balanced_accuracy':float(np.mean([r['train_metrics']['balanced_accuracy'] for r in success])) if success else None,
                'mean_dimensions':float(np.mean([r['dimensions'] for r in success])) if success else None,
                'failed_folds':len(rows)-len(success),'warning_count':sum(len(r['warnings']) for r in rows)})
        valid=[r for r in ranking if r['eligible']]
        if not valid: raise ValueError('No candidate completed every declared fold')
        # Predeclared tie break: lexicographic candidate ID; dummy remains eligible.
        winner=sorted(valid,key=lambda r:(-r['mean_balanced_accuracy'],r['candidate_id']))[0]
        pd.DataFrame(ranking).to_csv(self.staging/f'{stage}_ranking.csv',index=False)
        return winner,ranking,trials


def validate_config(c):
    positive_integer(c['bootstrap_repeats'], 'bootstrap_repeats')
    if c.get('budget_scope','per_invocation')!='per_invocation': raise ValueError('Supported budget_scope is per_invocation')
    if c['mode'] not in {'quick','nested'}: raise ValueError('mode must be quick or nested')
    if c['evaluation_mode']!='development_only': raise ValueError('Tuning runner accepts development_only; historical scoring is not part of this runner')
    if c['group_version']!=GROUP_VERSION or c['dataset_version']!=LEGACY_DATASET:
        raise ValueError('Comparable runner requires the frozen legacy dataset and artist-string protocol')
    if c.get('embeddings',False): raise ValueError('Audio-backed dataset/embedding manifest needed before enabling embeddings')
    if c['threshold_policy']!='estimator_default': raise ValueError('Only predeclared native thresholds are implemented')
    positive_integer(c['parallelism'],'parallelism')
    if c['parallelism']>4 or any(isinstance(c[k],bool) or not isinstance(c[k],(int,float)) or not np.isfinite(c[k]) or c[k]<=0 for k in ('timeout_seconds','fit_timeout_seconds')):
        raise ValueError('Invalid compute budget')
    specs=c['candidates']
    if len(specs)>c['max_trials'] or len({s['id'] for s in specs})!=len(specs): raise ValueError('Duplicate IDs or candidate budget exceeded')
    for family in {s['family'] for s in specs}:
        if sum(s['family']==family for s in specs)>c['max_trials_per_family']: raise ValueError('Per-family trial budget exceeded')
    if not any(s['family']=='dummy' for s in specs): raise ValueError('Dummy baseline is mandatory')


def invocation_deadline(config, previous):
    """Elapsed totals are telemetry; an explicit resume starts a new invocation budget."""
    return time.monotonic()+config['timeout_seconds']


def run(config_path,run_id,resume=False):
    require_supported_platform()
    config=json.loads(Path(config_path).read_text()); validate_config(config)
    destination=run_location(run_id)
    if destination.exists(): raise FileExistsError('Completed run is immutable; choose a new ID')
    ctx=load_dataset(ROOT/config['data'],dataset_version=config['dataset_version'],expected_sha256=config['data_sha256'])
    frame=ctx.frame; features=ctx.features
    split=pd.read_csv(ROOT/config['split_manifest'])
    if sha256_file(ROOT/config['split_manifest'])!=config['split_sha256']: raise ValueError('Frozen split manifest hash mismatch')
    if split.track_id.duplicated().any() or set(split.track_id)!=set(frame.track_id): raise ValueError('Invalid frozen split membership')
    aligned=split.set_index('track_id').loc[frame.track_id]
    if not np.array_equal(aligned.label,frame.label): raise ValueError('Frozen split labels changed')
    dev_ids=split.loc[split.partition=='train','track_id']; historical_ids=split.loc[split.partition=='test','track_id']
    if set(dev_ids)&set(historical_ids) or len(dev_ids)+len(historical_ids)!=len(frame): raise ValueError('Unknown/overlapping split partitions')
    development=frame.loc[frame.track_id.isin(dev_ids)].reset_index(drop=True)
    if set(development.artist.map(normalize_artist))&set(frame.loc[frame.track_id.isin(historical_ids)].artist.map(normalize_artist)):
        raise ValueError('Historical artist overlap')
    RUNS.mkdir(parents=True,exist_ok=True); staging=RUNS/f'.{run_id}.partial'
    code_hash=json_hash({str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'chorus_hit').glob('*.py'))})
    identity={'config_sha256':json_hash(config),'code_sha256':code_hash,'dataset_sha256':ctx.source_sha256}
    if staging.exists():
        if not resume: raise FileExistsError('Partial run exists; use --resume with identical code/config/data')
        previous=json.loads((staging/'state.json').read_text())
        if previous['identity']!=identity: raise ValueError('Resume identity changed; use a new run ID')
    else:
        if resume: raise FileNotFoundError('No partial run to resume')
        staging.mkdir(); previous={'elapsed_seconds':0,'attempts':0}
    started=time.monotonic(); deadline=invocation_deadline(config,previous)
    state={'run_id':run_id,'status':'running','identity':identity,'attempts':previous['attempts']+1,
        'elapsed_seconds':previous['elapsed_seconds'],'budget_scope':'per_invocation','invocation_budget_seconds':config['timeout_seconds'],'started_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(staging/'state.json',state); atomic_json(staging/'config.json',config)
    membership=split.copy(); membership['partition']=membership.partition.map({'train':'development','test':'historical_test'})
    membership.to_csv(staging/'membership.csv',index=False)
    print(f'{run_id}: {len(development)} development rows; {len(config["candidates"])} candidates; historical scoring disabled',flush=True)
    runner=Runner(staging,development,features,config,deadline); outer_predictions=[]; outer_rows=[]; all_rankings=[]
    specs={s['id']:s for s in config['candidates']}
    try:
        if config['mode']=='nested':
            folds=grouped_folds(development,config['outer_folds'],config['outer_seed'],development.artist.map(normalize_artist),features)
            atomic_json(staging/'outer_folds.json',[{'fold':i,'fit_track_ids':development.track_id.iloc[fit].tolist(),
                'validation_track_ids':development.track_id.iloc[val].tolist()} for i,(fit,val) in enumerate(folds)])
            family_predictions=[]
            for outer,(fit,val) in enumerate(folds):
                stage=f'outer{outer}'; winner,ranking,trials=runner.search(stage,fit,config['inner_seed'])
                all_rankings.extend(ranking)
                best_by_family={}
                for row in sorted([r for r in ranking if r['eligible']],key=lambda r:(-r['mean_balanced_accuracy'],r['candidate_id'])):
                    best_by_family.setdefault(row['family'],row)
                chosen={r['candidate_id'] for r in best_by_family.values()}|{winner['candidate_id']}
                tasks=[{'key':f'{stage}_assessment_{cid}','stage':f'{stage}_assessment','fold':outer,'spec':specs[cid],
                        'fit':fit.tolist(),'validation':val.tolist()} for cid in sorted(chosen)]
                assessed=runner.execute(tasks)
                for result in assessed:
                    if result['status']!='complete': raise ValueError('Outer assessment failed; procedure estimate incomplete')
                    for row in result['predictions']:
                        entry={**row,'fold':outer,'candidate_id':result['spec']['id'],'family':result['spec']['family']}
                        family_predictions.append(entry)
                        if result['spec']['id']==winner['candidate_id']: outer_predictions.append(entry)
                    if result['spec']['id']==winner['candidate_id']:
                        outer_rows.append({'fold':outer,'candidate_id':winner['candidate_id'],'inner_selection_score':winner['mean_balanced_accuracy'],
                            **result['metrics'],'dimensions':result['dimensions'],'train_balanced_accuracy':result['train_metrics']['balanced_accuracy']})
                print(f'Outer {outer+1}: selected {winner["candidate_id"]}; balanced accuracy {outer_rows[-1]["balanced_accuracy"]:.3f}',flush=True)
            pd.DataFrame(family_predictions).to_csv(staging/'family_outer_predictions.csv',index=False)
            atomic_json(staging/'outer_fold_metrics.json',outer_rows)
        winner,ranking,trials=runner.search('final_development',np.arange(len(development)),config['inner_seed'])
        all_rankings.extend(ranking)
        if config['mode']=='quick':
            for r in trials:
                if r['spec']['id']==winner['candidate_id']:
                    outer_predictions.extend({**p,'fold':r['fold'],'candidate_id':winner['candidate_id'],'family':r['spec']['family']} for p in r['predictions'])
        predictions=pd.DataFrame(outer_predictions)
        if predictions.track_id.duplicated().any() or set(predictions.track_id)!=set(development.track_id): raise ValueError('Incomplete or repeated development OOF records')
        # Different outer estimators have incomparable score scales: pooled AUC stays undefined.
        scores=predictions.score if config['mode']=='quick' else None
        metrics=classification_metrics(predictions.label,predictions.prediction,scores)
        if config['mode']=='nested': metrics['roc_auc_note']='Pooled ROC-AUC omitted: outer folds can select different score scales; see per-fold ROC-AUC.'
        evaluation='nested_development' if config['mode']=='nested' else 'tuning_development'
        ci=bootstrap_intervals(predictions.label,predictions.prediction,predictions.artist_group,scores,config['bootstrap_repeats'],config['seed'])
        selected=specs[winner['candidate_id']]
        with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); model=make_estimator(selected,features,config['seed']); model.fit(development[features],development.label)
        _,semantics=score_model(model,development[features].iloc[:1])
        # Learning curves are post-selection diagnostics, not an independent performance estimate.
        curve_tasks=[]; diagnostic_folds=grouped_folds(development,config['inner_folds'],config['inner_seed'],development.artist.map(normalize_artist),features)
        for fold,(fit,val) in enumerate(diagnostic_folds):
            g=development.artist.map(normalize_artist).to_numpy(); unique=sorted(set(g[fit]),key=lambda s:json_hash([config['seed'],s]))
            for fraction in config['learning_curve_fractions']:
                chosen_groups=unique[:max(2,int(np.ceil(len(unique)*fraction)))]; subset=fit[np.isin(g[fit],chosen_groups)]
                curve_tasks.append({'key':f'learning_{fold}_{fraction}','stage':f'learning_{fraction}','fold':fold,'spec':selected,
                                    'fit':subset.tolist(),'validation':val.tolist()})
        curves=runner.execute(curve_tasks)
        pd.DataFrame([{'fold':r['fold'],'stage':r['stage'],'fit_rows':r['fit_rows'],'fit_groups':r['fit_groups'],
            'status':r['status'],'train_balanced_accuracy':r.get('train_metrics',{}).get('balanced_accuracy'),
            'validation_balanced_accuracy':r.get('metrics',{}).get('balanced_accuracy'),'error':r.get('error')} for r in curves]).to_csv(staging/'learning_curves.csv',index=False)
        pd.DataFrame(all_rankings).to_csv(staging/'ablation_table.csv',index=False)
        predictions.to_csv(staging/'predictions.csv',index=False)
        group_rows=[{'artist_group':group,**classification_metrics(rows.label,rows.prediction)} for group,rows in predictions.groupby('artist_group')]
        atomic_json(staging/'per_group_metrics.json',group_rows)
        fold_values=[r['balanced_accuracy'] for r in outer_rows]
        evidence='candidate_not_validated'
        if evaluation=='nested_development' and ci['intervals']['balanced_accuracy'] is not None and ci['intervals']['balanced_accuracy'][0]<=.5: evidence='no_supported_improvement'
        summary={'run_id':run_id,'model_name':winner['candidate_id'],'candidate':selected,'feature_count':len(features),
            'retained_dimensions':int(model[:-1].transform(development[features].iloc[:1]).shape[1]),
            'evaluation_status':evaluation,'metrics':metrics,'ci95':ci['intervals'],'uncertainty':ci,
            'target_status':target_status(metrics,evaluation),'tuning_balanced_accuracy':winner['mean_balanced_accuracy'],
            'selection_metric':f'{config["inner_folds"]}-fold development balanced accuracy; predeclared ID tie break',
            'fold_mean_balanced_accuracy':float(np.mean(fold_values)) if fold_values else None,
            'fold_std_balanced_accuracy':float(np.std(fold_values)) if fold_values else None,
            'counts':{'train_songs':len(development),'train_artists':development.artist.map(normalize_artist).nunique(),
                      'evaluation_songs':len(predictions),'evaluation_artists':predictions.artist_group.nunique(),
                      'historical_songs_reserved':len(historical_ids)},'dataset_rows':len(frame),
            'evidence_status':evidence,'task_version':TASK_VERSION,'group_version':GROUP_VERSION,
            'label_definition':'1: source year-end Hot 100 hit; 0: other sampled weekly Hot 100 song',
            'historical_test_evaluated':False,'fresh_test_available':False,'final_fit_warnings':[str(w.message) for w in caught],
            'promotion':{'promoted':False,'decision':'Keep baseline demo. Candidate is for research; no fresh confirmation.'},
            'notes':['Thresholds are native estimator defaults; no calibration or threshold search.',
                     'Nested scores evaluate the declared selection procedure; final candidate uses full-development tuning.',
                     'Learning curves are retrospective diagnostics for the selected configuration.']}
        bundle={'pipeline':model,'run_id':run_id,'model_name':winner['candidate_id'],'features':features,
            'labels':{str(k):v for k,v in LABELS.items()},'extractor_version':EXTRACTOR_VERSION,'data_sha256':ctx.source_sha256,
            'train_track_ids':development.track_id.tolist(),'score_semantics':semantics}
        joblib.dump(bundle,staging/'model.joblib',compress=3)
        reloaded=joblib.load(staging/'model.joblib')
        if not np.array_equal(reloaded['pipeline'].predict(development[features]),model.predict(development[features])): raise ValueError('Reload prediction mismatch')
        atomic_json(staging/'summary.json',summary)
        state.update(status='complete',elapsed_seconds=state['elapsed_seconds']+time.monotonic()-started); atomic_json(staging/'state.json',state)
        manifest={'run_id':run_id,'dataset':{'path':config['data'],'dataset_version':config['dataset_version'],
                'source_sha256':ctx.source_sha256,'processed_fingerprint':ctx.processed_fingerprint},
            'raw_schema':features,'labels':bundle['labels'],'extractor_version':EXTRACTOR_VERSION,'score_semantics':semantics,
            'task_version':TASK_VERSION,'group_version':GROUP_VERSION,'model_file':'model.joblib','summary_file':'summary.json','predictions_file':'predictions.csv',
            'versions':{p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','librosa','joblib']},
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'dirty_tree':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            'code_sha256':code_hash,'config_sha256':identity['config_sha256'],'fit_population':'original development partition only',
            'evaluation_status':evaluation,'trial_budget':config['max_trials'],'elapsed_seconds':state['elapsed_seconds'],
            'threshold_policy':config['threshold_policy'],'historical_membership_hash':config['split_sha256']}
        finalize_run(staging,manifest,destination)
        print(json.dumps({'run_id':run_id,'selected':winner['candidate_id'],'evaluation':evaluation,'metrics':metrics,'seconds':state['elapsed_seconds']}),flush=True)
        return destination
    except BaseException as exc:
        state.update(status='incomplete',elapsed_seconds=state['elapsed_seconds']+time.monotonic()-started,error=f'{type(exc).__name__}: {exc}')
        atomic_json(staging/'state.json',state)
        (staging/'error.log').write_text(traceback.format_exc())
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run-id',required=True); p.add_argument('--resume',action='store_true')
    p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    require_supported_platform()
    if a._worker:
        run(a.config,a.run_id,a.resume); return
    # An isolated process group also bounds final refitting/bootstrapping, not only CV fits.
    config=json.loads(a.config.read_text()); validate_config(config)
    command=[sys.executable,'-m','chorus_hit.train_v2','--config',str(a.config.resolve()),'--run-id',a.run_id,'--_worker']
    if a.resume: command.append('--resume')
    try:
        return_code=supervise(command,config['timeout_seconds'],ROOT)
    except subprocess.TimeoutExpired:
        partial=RUNS/f'.{a.run_id}.partial'; state_path=partial/'state.json'
        if state_path.exists():
            state=json.loads(state_path.read_text()); state.update(status='incomplete',error='Supervised wall-time budget exhausted')
            atomic_json(state_path,state)
        print('Wall-time budget exhausted; completed checkpoints preserved.',file=sys.stderr)
        return_code=2
    raise SystemExit(return_code)


if __name__=='__main__': main()
