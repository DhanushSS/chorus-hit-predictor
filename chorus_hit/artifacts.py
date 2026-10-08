"""One validated artifact contract for the app, CLI, and report builders."""
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import os
from hashlib import sha256
from zipfile import ZipFile, BadZipFile

import joblib
import numpy as np
import pandas as pd

from .audit import atomic_json, sha256_file
from .config import ROOT
from .evaluation import score_model

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
    expected_trials={name:digest for name,digest in manifest['files'].items() if name.startswith('trials/')}
    archive=path/'trials.zip'
    if archive.exists():
        if not archive.is_file() or not expected_trials: raise ValueError('Unexpected trial archive')
        try:
            with ZipFile(archive) as saved:
                names=saved.namelist()
                if len(names)!=len(expected_trials) or set(names)!=set(expected_trials):
                    raise ValueError('Trial archive entries do not match the run manifest')
                for name,digest in expected_trials.items():
                    if sha256(saved.read(name)).hexdigest()!=digest:
                        raise ValueError(f'Missing or altered run asset: {name}')
        except (BadZipFile, OSError) as error:
            raise ValueError('Corrupt trial archive') from error
    for file,digest in manifest['files'].items():
        p=(path/file).resolve()
        if not p.is_relative_to(path.resolve()): raise ValueError('Artifact path escapes run directory')
        if archive.exists() and file in expected_trials:
            if p.exists(): raise ValueError(f'Duplicate trial asset outside archive: {file}')
            continue
        if not p.is_file() or sha256_file(p)!=digest: raise ValueError(f'Missing or altered run asset: {file}')
    for key in ['model_file','summary_file']:
        if manifest[key] not in manifest['files']: raise ValueError(f'Unverified {key}')


def trial_records(path):
    """Read saved trials from the compact archive or an unarchived run."""
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text())
    verify_files(path,manifest)
    archive=path/'trials.zip'
    if archive.is_file():
        with ZipFile(archive) as saved:
            return [json.loads(saved.read(name)) for name in sorted(saved.namelist())]
    return [json.loads(p.read_text()) for p in sorted((path/'trials').glob('*.json'))]


@dataclass
class RunAssets:
    path: Path
    manifest: dict
    bundle: dict
    summary: dict
    data: pd.DataFrame
    predictions: pd.DataFrame | None

    def validate_input(self,X,extractor_version=None):
        if len(X)==0: raise ValueError('Input must contain at least one row')
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
    manifest=json.loads((path/'manifest.json').read_text())
    from .contracts import require_fields, validate_evaluation
    require_fields(manifest, ['run_id','status','files','versions','dataset','model_file','summary_file','predictions_file','task_version','group_version','raw_schema','labels','extractor_version','score_semantics'], 'manifest')
    verify_files(path,manifest)
    for package,pin in manifest['versions'].items():
        if version(package)!=pin: raise ValueError(f'Package mismatch for {package}: requires {pin}')
    # Load only the locally generated model after its bytes and environment were checked.
    bundle=joblib.load(path/manifest['model_file']); summary=json.loads((path/manifest['summary_file']).read_text())
    require_fields(summary, ['run_id','evaluation_status','metrics','target_status','task_version','group_version','feature_count','model_name'], 'summary')
    require_fields(bundle, ['run_id','features','labels','extractor_version','data_sha256','pipeline','score_semantics','train_track_ids','model_name'], 'model bundle')
    require_fields(manifest['dataset'], ['path','dataset_version','source_sha256'], 'dataset')
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
    if summary['task_version']!=manifest['task_version'] or summary['group_version']!=manifest['group_version']:
        raise ValueError('Summary task/group contract mismatch')
    if summary['feature_count']!=len(bundle['features']) or summary['model_name']!=bundle['model_name']:
        raise ValueError('Summary/model mismatch')
    if predictions is None: raise ValueError('Saved evaluation predictions required')
    validate_evaluation(path,manifest,summary,bundle,data,predictions,
                        trial_records(path) if summary['evaluation_status']!='historical_test' else [])
    return RunAssets(path,manifest,bundle,summary,data,predictions)


def finalize_run(staging,manifest,destination):
    staging=Path(staging); destination=Path(destination)
    if destination.exists(): raise FileExistsError('Completed run already exists; choose a new run ID')
    manifest['status']='complete'
    manifest['files']={str(p.relative_to(staging)):sha256_file(p) for p in sorted(staging.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    atomic_json(staging/'manifest.json',manifest)
    verify_files(staging,manifest)
    os.rename(staging,destination)



if __name__=='__main__':
    print(load_run().summary['run_id'])
