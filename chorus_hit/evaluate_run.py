"""Deliberate one-time historical comparison of a frozen development candidate."""
import argparse,json,os,tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from .artifacts import load_run
from .audit import atomic_json,sha256_file
from .config import ROOT
from .data import normalize_artist
from .evaluation import classification_metrics,bootstrap_intervals,paired_bootstrap,target_status


def evaluate_historical(run_id,out):
    a=load_run(run_id);baseline=load_run('v1_baseline');out=Path(out)
    if out.exists():raise FileExistsError('Historical evaluation output exists; preserve it')
    if a.summary['evaluation_status'] not in {'tuning_development','nested_development'}:raise ValueError('Freeze a completed development run first')
    pred0=baseline.predictions.set_index('track_id');ids=pred0.index.tolist()
    if set(ids)&set(a.bundle['train_track_ids']):raise ValueError('Historical rows entered model fitting')
    data=a.data.set_index('track_id').loc[ids]
    train=a.data.loc[a.data.track_id.isin(a.bundle['train_track_ids'])]
    if set(data.artist.map(normalize_artist))&set(train.artist.map(normalize_artist)):raise ValueError('Historical artist overlap')
    prediction,score,semantics=a.predict(data[a.bundle['features']]); m=classification_metrics(data.label,prediction,score)
    groups=data.artist.map(normalize_artist).to_numpy()
    records=data[['artist','title','label']].reset_index();records['artist_group']=groups;records['prediction']=prediction;records['score']=score
    uncertainty=bootstrap_intervals(data.label,prediction,groups,score)
    summary={'run_id':run_id,'evaluation_status':'historical_test','model_name':a.bundle['model_name'],'metrics':m,
        'ci95':uncertainty['intervals'],'uncertainty':uncertainty,'score_semantics':semantics,
        'target_status':target_status(m,'historical_test'),'paired_vs_v1':paired_bootstrap(data.label,prediction,pred0.prediction,groups),
        'groups':len(set(groups)),'task_version':a.manifest['task_version'],'dataset_version':a.manifest['dataset']['dataset_version'],
        'group_version':a.manifest['group_version'],'fresh_test_available':False,
        'note':'Known historical benchmark scored after development selection. No model/threshold/promotion decisions use these labels.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.historical-',dir=out.parent) as tmp:
        staging=Path(tmp)/'evaluation';staging.mkdir();records.to_csv(staging/'predictions.csv',index=False);atomic_json(staging/'summary.json',summary)
        atomic_json(staging/'manifest.json',{'status':'complete','source_run':run_id,'source_run_manifest_sha256':sha256_file(a.path/'manifest.json'),
            'baseline_manifest_sha256':sha256_file(baseline.path/'manifest.json'),'files':{p.name:sha256_file(p) for p in staging.iterdir() if p.is_file()}})
        os.rename(staging,out)
    return summary


def load_historical(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text()); a=load_run(m['source_run'])
    if m['status']!='complete' or m['source_run_manifest_sha256']!=sha256_file(a.path/'manifest.json'):raise ValueError('Historical run identity mismatch')
    if m['baseline_manifest_sha256']!=sha256_file(load_run('v1_baseline').path/'manifest.json'):raise ValueError('Historical baseline identity mismatch')
    for name,digest in m['files'].items():
        f=(path/name).resolve()
        if not f.is_relative_to(path.resolve()) or sha256_file(f)!=digest:raise ValueError('Historical artifact mismatch')
    s=json.loads((path/'summary.json').read_text());pred=pd.read_csv(path/'predictions.csv')
    if s['run_id']!=m['source_run'] or s['evaluation_status']!='historical_test':raise ValueError('Historical summary mismatch')
    for k,v in classification_metrics(pred.label,pred.prediction,pred.score).items():
        if s['metrics'][k]!=v:raise ValueError(f'Historical metric mismatch: {k}')
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    p.add_argument('--historical',action='store_true',required=True,help='Explicitly score the already known historical benchmark')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(evaluate_historical(a.run_id,a.out),indent=2))

if __name__=='__main__':main()
