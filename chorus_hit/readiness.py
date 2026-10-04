"""Read-only environment, saved-run and evidence readiness check."""
import argparse
import hashlib
from collections import Counter
from importlib.metadata import version, PackageNotFoundError
import json
import platform
from pathlib import Path
import sys
from .config import ROOT


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_audit(root=ROOT):
    from .ingestion import record_validation
    root=Path(root); source=root/'data/recording_audit_v1.json'
    records=json.loads(source.read_text())['records']
    from .data import load_data
    data=load_data(root/'data/chorus_features.csv')
    if len(records)!=len(data) or {r['recording_id'] for r in records}!=set(data.track_id): raise ValueError('Audit/data membership mismatch')
    labels=data.set_index('track_id').label
    if any(r['label']!=labels.loc[r['recording_id']] for r in records): raise ValueError('Audit labels mismatch')
    counts=Counter(r.get('label_evidence_status','unknown') for r in records)
    structural=[record_validation(r) for r in records]
    verified=[r for r in records if r.get('label_evidence_status')=='verified' and
              isinstance(r.get('label_source'),str) and r['label_source'].startswith(('https://','http://')) and
              all(r.get(k) for k in ['chart_market','chart_type','observation_start','observation_end','recording_version'])]
    return {'records':len(records),'source_sha256':sha256_file(source),'dataset_sha256':sha256_file(root/'data/chorus_features.csv'),
            'label_status_counts':dict(counts),'inherited':counts['inherited_unverified'],
            'verified_with_references':len(verified),'disputed':counts['disputed'],
            'unknown_or_unsupported':len(records)-counts['inherited_unverified']-counts['disputed']-len(verified),
            'canonical_identity_supported':sum(r['identity_verified'] for r in structural),
            'unresolved_recordings':sum(not r['identity_verified'] for r in structural),
            'strict_audio_study_eligible':sum(r['eligible_for_audio_study'] for r in structural),
            'source_references':sorted({r['label_source'] for r in records if r.get('label_source')}),
            'original_ids_preserved':True,'independent_chart_review_performed':False,
            'duplicate_resolution':'not established; no inferred canonical IDs',
            'schema_validated':True,'waveform_parity_verified':False,
            'real_audio_selector_review':{'reviewed_recordings':0,'annotation_tolerance_seconds':None,'accuracy':None},
            'fresh_test_available':False}


def check_readiness(root=ROOT):
    root=Path(root); errors=[]; packages={}
    if sys.version_info[:2]!=(3,12): errors.append('Supported Python is 3.12')
    for line in (root/'requirements.txt').read_text().splitlines():
        if '==' not in line or line.startswith('#'):continue
        package,pin=line.strip().split('==')
        try: installed=version(package)
        except PackageNotFoundError: installed=None
        packages[package]={'required':pin,'installed':installed,'compatible':installed==pin}
        if installed!=pin:errors.append(f'{package}: requires {pin}, installed {installed}')
    runs={};active=None
    try:
        from .artifacts import load_run
    except ImportError as exc:
        load_run=None
        errors.append(f'Model dependencies unavailable: {exc}')
    try:active=json.loads((root/'configs/active_run.json').read_text())['run_id']
    except Exception as exc:errors.append(f'Active run: {exc}')
    for p in sorted((root/'results/v2').glob('*/manifest.json')):
        if p.parent.name.startswith('.'):continue
        try:
            if load_run is None: raise ValueError('Install pinned dependencies before loading models')
            a=load_run(root=p.parent)
            runs[p.parent.name]={'status':'valid','evaluation_status':a.summary['evaluation_status'],
                                 'evidence_status':a.summary['evidence_status'],'balanced_accuracy':a.summary['metrics']['balanced_accuracy']}
        except Exception as exc:runs[p.parent.name]={'status':'invalid','error':str(exc)};errors.append(f'{p.parent.name}: {exc}')
    if active not in runs:errors.append('Active run is missing')
    try:evidence=evidence_audit(root)
    except Exception as exc:evidence={'error':str(exc)};errors.append(f'Evidence audit: {exc}')
    return {'software_ready':not errors,'python':platform.python_version(),'platform':platform.platform(),
            'supervised_training_supported':sys.platform in {'darwin','linux'},'packages':packages,
            'active_run':active,'runs':runs,'evidence':evidence,'errors':errors,
            'limits':['Software checks do not establish musical accuracy on uploads.','Faculty approval is recorded in docs/task_alignment.md.','Actual review-device rehearsal is a separate manual check.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence-only',action='store_true');p.add_argument('--out',type=Path)
    args=p.parse_args();report=evidence_audit() if args.evidence_only else check_readiness()
    if args.out:
        # Preserve earlier readiness evidence rather than replacing it.
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
    if not args.evidence_only and not report['software_ready']:raise SystemExit(1)


if __name__=='__main__':main()
