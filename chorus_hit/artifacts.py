"""One validated artifact contract for the app, CLI, and report builders."""
import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import os

import joblib
import numpy as np
import pandas as pd

from .audit import atomic_json, sha256_file, LEGACY_DATASET, TASK_VERSION, GROUP_VERSION
from .config import ROOT, FEATURE_COLUMNS, LABELS
from .evaluation import classification_metrics, target_status, score_model

RUNS=ROOT/'results/v2'
ACTIVE=ROOT/'configs/active_run.json'
EXTRACTOR_VERSION='legacy-librosa-518-v1'


def run_location(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Invalid run ID')
    return RUNS/run_id


def active_run():
    return json.loads(ACTIVE.read_text())['run_id']


def run_cache_key(run_id):
    # Verification is cheap and occurs outside Streamlit's cached loader.
    path=run_location(run_id); manifest=json.loads((path/'manifest.json').read_text())
    verify_files(path,manifest)
    dataset=ROOT/manifest['dataset']['path']
    if sha256_file(dataset)!=manifest['dataset']['source_sha256']: raise ValueError('Dataset hash mismatch')
    return run_id,sha256_file(path/'manifest.json')


def verify_files(path,manifest):
    if manifest['status']!='complete': raise ValueError('Run is incomplete and cannot serve predictions')
    for file,digest in manifest['files'].items():
        p=(path/file).resolve()
        if not p.is_relative_to(path.resolve()): raise ValueError('Artifact path escapes run directory')
        if not p.is_file() or sha256_file(p)!=digest: raise ValueError(f'Missing or altered run asset: {file}')
    for key in ['model_file','summary_file']:
        if manifest[key] not in manifest['files']: raise ValueError(f'Unverified {key}')


@dataclass
class RunAssets:
    path: Path
    manifest: dict
    bundle: dict
    summary: dict
    data: pd.DataFrame
    predictions: pd.DataFrame | None

    def validate_input(self,X,extractor_version=None):
        if X.columns.tolist()!=self.bundle['features']: raise ValueError('Input feature schema/order mismatch')
        if not np.isfinite(X.to_numpy(dtype=float)).all(): raise ValueError('Input contains nonfinite features')
        if extractor_version is not None and extractor_version!=self.bundle['extractor_version']:
            raise ValueError('Upload extractor/model contract mismatch')

    def predict(self,X,extractor_version=None):
        self.validate_input(X,extractor_version)
        score,semantics=score_model(self.bundle['pipeline'],X)
        if semantics!=self.bundle['score_semantics']: raise ValueError('Model score semantics mismatch')
        return self.bundle['pipeline'].predict(X),score,semantics


def load_run(run_id=None, *, root=None):
    path=Path(root) if root is not None else run_location(run_id or active_run())
    manifest=json.loads((path/'manifest.json').read_text()); verify_files(path,manifest)
    for package,pin in manifest['versions'].items():
        if version(package)!=pin: raise ValueError(f'Package mismatch for {package}: requires {pin}')
    # Load only the locally generated model after its bytes and environment were checked.
    bundle=joblib.load(path/manifest['model_file']); summary=json.loads((path/manifest['summary_file']).read_text())
    if not (manifest['run_id']==bundle['run_id']==summary['run_id']): raise ValueError('Mixed run identities')
    if bundle['features']!=manifest['raw_schema'] or bundle['labels']!=manifest['labels']:
        raise ValueError('Model schema/label contract mismatch')
    if bundle['extractor_version']!=manifest['extractor_version'] or bundle['data_sha256']!=manifest['dataset']['source_sha256']:
        raise ValueError('Model dataset/extractor contract mismatch')
    if bundle['pipeline'].feature_names_in_.tolist()!=bundle['features']: raise ValueError('Fitted model input order mismatch')
    if bundle['score_semantics']!=manifest['score_semantics']: raise ValueError('Score contract mismatch')
    from .audit import load_dataset
    context=load_dataset(ROOT/manifest['dataset']['path'],features=manifest['raw_schema'],
        dataset_version=manifest['dataset']['dataset_version'],expected_sha256=manifest['dataset']['source_sha256'])
    data=context.frame
    train=bundle['train_track_ids']
    if len(train)!=len(set(train)) or not set(train).issubset(data.track_id): raise ValueError('Invalid training membership')
    predictions=None
    if manifest.get('predictions_file'):
        if manifest['predictions_file'] not in manifest['files']: raise ValueError('Unverified predictions')
        predictions=pd.read_csv(path/manifest['predictions_file'])
        if predictions.track_id.duplicated().any(): raise ValueError('Duplicated evaluation rows')
        actual=data.set_index('track_id').loc[predictions.track_id]
        if not np.array_equal(actual.label,predictions.label): raise ValueError('Prediction labels are misaligned')
        if summary['evaluation_status']=='historical_test':
            if set(predictions.track_id)&set(train): raise ValueError('Historical evaluation overlaps training')
            from .data import normalize_artist
            if set(actual.artist.map(normalize_artist))&set(data.loc[data.track_id.isin(train)].artist.map(normalize_artist)):
                raise ValueError('Historical artist overlap')
            expected=manifest.get('evaluation_track_ids',[])
            if set(predictions.track_id)!=set(expected): raise ValueError('Evaluation membership mismatch')
    if summary['task_version']!=manifest['task_version'] or summary['group_version']!=manifest['group_version']:
        raise ValueError('Summary task/group contract mismatch')
    if predictions is not None:
        scores=None if summary['evaluation_status']=='nested_development' else predictions.score
        calculated=classification_metrics(predictions.label,predictions.prediction,scores)
        for metric in ['accuracy','balanced_accuracy','precision','recall','f1','class_0_recall','macro_f1','roc_auc']:
            actual=calculated[metric]; reported=summary['metrics'][metric]
            if actual is None and reported is None: continue
            if actual is None or reported is None or not np.isclose(actual,reported,rtol=0,atol=1e-12):
                raise ValueError(f'Summary metric mismatch: {metric}')
        if summary['target_status']!=target_status(summary['metrics'],summary['evaluation_status']):
            raise ValueError('Target check mismatch')
    if summary['feature_count']!=len(bundle['features']) or summary['model_name']!=bundle['model_name']:
        raise ValueError('Summary/model mismatch')
    return RunAssets(path,manifest,bundle,summary,data,predictions)


def finalize_run(staging,manifest,destination):
    staging=Path(staging); destination=Path(destination)
    if destination.exists(): raise FileExistsError('Completed run already exists; choose a new run ID')
    manifest['status']='complete'
    manifest['files']={str(p.relative_to(staging)):sha256_file(p) for p in sorted(staging.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    atomic_json(staging/'manifest.json',manifest)
    verify_files(staging,manifest)
    os.rename(staging,destination)


def preserve_legacy():
    run_id='v1_baseline'; dest=run_location(run_id)
    if dest.exists(): return load_run(run_id)
    RUNS.mkdir(parents=True,exist_ok=True); staging=RUNS/'.v1_baseline.partial'; staging.mkdir(exist_ok=False)
    old=json.loads((ROOT/'results/metrics.json').read_text()); b=joblib.load(ROOT/'models/selected_model.joblib')
    pred=pd.read_csv(ROOT/'results/test_predictions.csv'); m=classification_metrics(pred.label,pred.prediction,pred.score)
    b.update(run_id=run_id,labels={str(k):v for k,v in LABELS.items()},extractor_version=EXTRACTOR_VERSION)
    from .data import load_data
    _,semantics=score_model(b['pipeline'],load_data()[FEATURE_COLUMNS].head(1)); b['score_semantics']=semantics
    row=next(x for x in old['models'] if x['selected'])
    summary={'run_id':run_id,'model_name':b['model_name'],'feature_count':len(b['features']),
        'evaluation_status':'historical_test','metrics':m,'ci95':old['selected_test_ci95_artist_bootstrap'],
        'target_status':target_status(m,'historical_test'),'tuning_balanced_accuracy':row['cv_balanced_accuracy'],
        'counts':old['split'],'dataset_rows':old['data_audit']['rows'],'evidence_status':'candidate_not_validated',
        'selection_metric':old['selection_metric'],'task_version':TASK_VERSION,'group_version':GROUP_VERSION,
        'label_definition':old['label_definition'],'historical_models':old['models'],
        'notes':['Legacy model selected before its original holdout evaluation.','Old test is now a historical benchmark, not fresh confirmation.']}
    joblib.dump(b,staging/'model.joblib',compress=3); atomic_json(staging/'summary.json',summary)
    shutil.copy(ROOT/'results/test_predictions.csv',staging/'predictions.csv')
    shutil.copy(ROOT/'results/split_manifest.csv',staging/'split_manifest.csv')
    shutil.copytree(ROOT/'results/figures',staging/'figures')
    manifest={'run_id':run_id,'dataset':{'path':'data/chorus_features.csv','source_sha256':b['data_sha256'],'dataset_version':LEGACY_DATASET},
        'raw_schema':b['features'],'labels':b['labels'],'extractor_version':EXTRACTOR_VERSION,
        'model_file':'model.joblib','summary_file':'summary.json','predictions_file':'predictions.csv',
        'score_semantics':semantics,'versions':b['versions'],'task_version':TASK_VERSION,'group_version':GROUP_VERSION,
        'evaluation_track_ids':pred.track_id.tolist(),'baseline_commit':'0e9e4628d0a72af79694bc555d64a562e83dcfa2'}
    finalize_run(staging,manifest,dest)
    if not ACTIVE.exists(): atomic_json(ACTIVE,{'run_id':run_id,'decision':'Preserve baseline demo; new candidates need development evidence before promotion'})
    return load_run(run_id)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--preserve-legacy',action='store_true')
    a=p.parse_args()
    if a.preserve_legacy: print(preserve_legacy().path)
    else: print(load_run().summary['run_id'])
