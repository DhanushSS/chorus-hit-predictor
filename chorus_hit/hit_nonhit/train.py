"""Resumable, bounded nested selection on development only; test labels are never loaded."""
from pathlib import Path
import time
import joblib
import numpy as np
from chorus_hit.config import FEATURE_COLUMNS, ROOT
from . import EXTRACTOR, GROUP_VERSION, LABEL_VERSION, TASK
from .audio import contract
from .common import (cli, code_identity, digest, environment, parser, read, runtime_check,
                     seal, sha, write)
from .freeze_split import grouped_folds, load_development
from .modeling import cluster_interval, estimator, metrics, score_values
from .schema import require


def fit(model, X, y, seconds):
    """Hard wall-clock supervision also stops fits stuck inside native code."""
    import os
    import subprocess
    import sys
    import tempfile
    from chorus_hit.supervision import stop_process_group
    with tempfile.TemporaryDirectory(prefix='hit_nonhit_fit_') as directory:
        folder = Path(directory)
        joblib.dump({'model': model, 'X': X, 'y': y}, folder / 'request.joblib')
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        command = [sys.executable, '-m', 'chorus_hit.hit_nonhit.fit_worker', '--request', str(folder / 'request.joblib'), '--response', str(folder / 'response.joblib')]
        with (folder / 'worker.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                try:
                    code = child.wait(timeout=seconds)
                except subprocess.TimeoutExpired as error:
                    raise TimeoutError('Predeclared per-fit time limit exceeded') from error
            finally:
                stop_process_group(child)
        if code:
            raise ValueError('Fit worker failed: ' + (folder / 'worker.log').read_text()[-2000:])
        result = joblib.load(folder / 'response.joblib')
        model.__dict__.clear()
        model.__dict__.update(result['model'].__dict__)
        return result['seconds'], result['warnings']


def candidate_config(cfg, synthetic, groups):
    values = cfg['candidates']
    require(1 <= len(values) <= 32, 'Candidate budget must be 1..32')
    require(len({c['id'] for c in values}) == len(values), 'Duplicate candidate IDs')
    require(any(c['family'] == 'dummy' for c in values), 'Dummy baseline required')
    if cfg.get('model_profile') == 'academic_small_sample_v1' and len(set(groups)) < 15:
        values = [c for c in values if c['family'] not in {'mlp', 'hist_boosting'}]
    if not synthetic:
        require({'dummy', 'logistic', 'linear_svm', 'rbf_svm', 'random_forest', 'extra_trees'} <= {c['family'] for c in values}, 'Mandatory baseline families missing')
    for c in values:
        require(c['id'].replace('_', '').isalnum(), 'Unsafe candidate ID')
        if c['family'] in {'mlp', 'hist_boosting'}:
            minimum = 15 if cfg.get('model_profile') == 'academic_small_sample_v1' else 50
            require(len(set(groups)) >= minimum, f'Conditional neural/boosting models need >={minimum} development groups')
        estimator(c, cfg['evaluation']['seed'])
    require(type(cfg['permutation_repeats']) is int and 1 <= cfg['permutation_repeats'] <= 10, 'Permutation budget must be 1..10')
    require(0 < cfg['per_fit_seconds'] <= 60 and 0 < cfg['total_fit_budget_seconds'] <= 28800, 'Compute budget outside supported bounds')
    return values


def diagnostics(X, y, groups, rows, cfg, reference):
    """Fixed development-only nuisance and within-group label-permutation checks."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    folds = grouped_folds(y, groups, cfg['evaluation']['inner_folds'], cfg['evaluation']['seed'])
    nuisance = pd.DataFrame([{'source': r['audio_source_type'], 'codec': r['audio_original_format'],
                             'year': int(r['first_release_date'][:4]), 'duration': r['recording_duration'],
                             'sample_rate': r['original_sample_rate'], 'channels': r['original_channels']} for r in rows])
    prediction = np.empty(len(y), dtype=int)
    for tr, va in folds:
        model = Pipeline([('input', ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['source', 'codec']),
            ('numeric', StandardScaler(), ['year', 'duration', 'sample_rate', 'channels'])])),
            ('model', LogisticRegression(C=.1, class_weight='balanced', max_iter=1000))])
        fit(model, nuisance.iloc[tr], y[tr], cfg['per_fit_seconds'])
        prediction[va] = model.predict(nuisance.iloc[va])
    nuisance_score = metrics(y, prediction)
    rng = np.random.default_rng(cfg['evaluation']['seed']); permuted_scores = []
    nontrivial = next((c for c in cfg['candidates'] if c['family'] == 'logistic'), cfg['candidates'][0])
    exchangeable = sum(len(set(y[groups == g])) == 2 for g in set(groups))
    for _ in range(cfg['permutation_repeats'] if exchangeable else 0):
        shuffled = y.copy()
        for group in sorted(set(groups)):
            indexes = np.flatnonzero(groups == group)
            shuffled[indexes] = rng.permutation(y[indexes])
        pred = np.empty(len(y), dtype=int)
        for tr, va in folds:
            model = estimator(nontrivial, cfg['evaluation']['seed'])
            fit(model, X.iloc[tr], shuffled[tr], cfg['per_fit_seconds'])
            pred[va] = model.predict(X.iloc[va])
        permuted_scores.append(metrics(shuffled, pred)['balanced_accuracy'])
    flags = []
    if nuisance_score['balanced_accuracy'] >= .65:
        flags.append('nuisance_only_balanced_accuracy_at_least_0.65')
    if permuted_scores and np.mean(permuted_scores) >= max(.6, reference - .02):
        flags.append('suspicious_permuted_label_performance')
    if not exchangeable:
        flags.append('within_group_permutation_not_identifiable')
    return {'nuisance_only': nuisance_score, 'within_group_permutations': permuted_scores,
            'permutation_exchangeable_groups': exchangeable, 'review_flags': flags,
            'feature_distributions': {'overall_mean': X.mean().to_dict(),
                                      'class_means': {str(k): X.loc[y == k].mean().to_dict() for k in [0, 1]}},
            'scope': 'development only; confounding warnings block locked evaluation pending a revised preregistered cohort'}


def train(split, destination, resume=False, dry_run=False):
    runtime_check(); split = Path(split)
    frame, sm, cfg = load_development(split)
    require(sm['extractor'] == contract() and sm['pinned_env'] == environment(), 'Split extractor/environment differs')
    require(not (split / 'test_access.json').exists(), 'Locked test already accessed; further selection prohibited')
    X = frame[FEATURE_COLUMNS]; y = frame.label.to_numpy(); groups = frame.group.to_numpy()
    cs = candidate_config(cfg, sm['synthetic'], groups); e = cfg['evaluation']
    outer = grouped_folds(y, groups, e['outer_folds'], e['seed'])
    maximum_fits = len(cs) * e['inner_folds'] * (e['outer_folds'] + 1) + e['outer_folds'] + 1 + e['inner_folds'] * (cfg['permutation_repeats'] + 1)
    require(maximum_fits * cfg['per_fit_seconds'] <= cfg['total_fit_budget_seconds'], 'Declared total budget cannot cover bounded fit count')
    out = Path(destination).resolve()
    if out.is_relative_to(ROOT):
        allowed = ROOT / ('output/hit_nonhit' if sm['synthetic'] else 'results/hit_nonhit/v1')
        require(out.is_relative_to(allowed), 'Wrong task output namespace (synthetic artifacts cannot be real results)')
    if dry_run:
        return {'synthetic': sm['synthetic'], 'candidates': len(cs), 'development_rows': len(y), 'test_labels_loaded': False}
    require(not (out / 'manifest.json').exists(), 'Completed run cannot be overwritten')
    require(resume or not out.exists(), 'Existing run needs --resume')
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.training.lock'
    with lock.open('x'):
        pass
    try:
        identity = {'split_manifest_hash': sha(split / 'manifest.json'), 'config_hash': digest(cfg),
                    'code': code_identity(), 'environment': environment(), 'synthetic': sm['synthetic']}
        if (out / 'resume_identity.json').exists():
            require(read(out / 'resume_identity.json') == identity, 'Cannot resume after data/config/code/environment changes')
        else:
            write(out / 'resume_identity.json', identity)
        checkpoints = out / 'trials'; checkpoints.mkdir(exist_ok=True)
        def trial(candidate, tr, va, scope):
            key = digest([candidate, frame.recording_id.iloc[tr].tolist(), frame.recording_id.iloc[va].tolist(), scope])
            path = checkpoints / (key + '.json')
            if path.exists():
                cached = read(path)
                payload = {k: v for k, v in cached.items() if k != 'payload_sha256'}
                require(cached.get('payload_sha256') == digest(payload), 'Checkpoint hash mismatch')
                require(cached['candidate'] == candidate and cached['scope'] == scope and cached['train_ids'] == frame.recording_id.iloc[tr].tolist() and cached['validation_ids'] == frame.recording_id.iloc[va].tolist(), 'Checkpoint membership mismatch')
                return cached
            spent = sum(read(p).get('seconds', 0) for p in checkpoints.glob('*.json'))
            require(spent < cfg['total_fit_budget_seconds'], 'Total predeclared fit budget exhausted; checkpoints preserved')
            record = {'candidate': candidate, 'scope': scope, 'train_ids': frame.recording_id.iloc[tr].tolist(),
                      'validation_ids': frame.recording_id.iloc[va].tolist(), 'status': 'failed'}
            began = time.perf_counter()
            try:
                model = estimator(candidate, e['seed'])
                seconds, ws = fit(model, X.iloc[tr], y[tr], cfg['per_fit_seconds'])
                record.update(status='complete', train_metrics=metrics(y[tr], model.predict(X.iloc[tr])),
                              validation_metrics=metrics(y[va], model.predict(X.iloc[va])), warnings=ws)
            except (ValueError, RuntimeError, TimeoutError) as error:
                record['error'] = str(error)
            record['seconds'] = time.perf_counter() - began
            record['payload_sha256'] = digest(record)
            write(path, record)
            return record
        def select(index, scope):
            inner = grouped_folds(y[index], groups[index], e['inner_folds'], e['seed'])
            scores = []
            for c in cs:
                records = [trial(c, index[tr], index[va], f'{scope}/inner_{i}') for i, (tr, va) in enumerate(inner)]
                good = all(r['status'] == 'complete' for r in records)
                scores.append({'candidate_id': c['id'], 'balanced_accuracy': float(np.mean([r['validation_metrics']['balanced_accuracy'] for r in records])) if good else None})
            usable = [s for s in scores if s['balanced_accuracy'] is not None]
            require(usable, 'All candidates failed')
            best = sorted(usable, key=lambda s: (-s['balanced_accuracy'], s['candidate_id']))[0]
            return next(c for c in cs if c['id'] == best['candidate_id']), scores
        oof = np.empty(len(y), dtype=int); outer_records = []
        for fold, (tr, va) in enumerate(outer):
            selected, scores = select(tr, f'outer_{fold}')
            saved = out / f'outer_{fold}.json'
            if saved.exists():
                r = read(saved)
                require(r.get('payload_sha256') == digest({k: v for k, v in r.items() if k != 'payload_sha256'}), 'Outer checkpoint hash mismatch')
                require(r['candidate'] == selected and r['validation_ids'] == frame.recording_id.iloc[va].tolist(), 'Outer checkpoint mismatch')
            else:
                model = estimator(selected, e['seed'])
                seconds, ws = fit(model, X.iloc[tr], y[tr], cfg['per_fit_seconds'])
                r = {'fold': fold, 'candidate': selected, 'selection': scores, 'train_ids': frame.recording_id.iloc[tr].tolist(),
                     'validation_ids': frame.recording_id.iloc[va].tolist(), 'predictions': model.predict(X.iloc[va]).tolist(),
                     'warnings': ws, 'seconds': seconds}
                r['payload_sha256'] = digest(r)
                write(saved, r)
            oof[va] = r['predictions']; outer_records.append(r)
        selected, comparison = select(np.arange(len(y)), 'final_development_selection')
        nested = metrics(y, oof)
        diagpath = out / 'diagnostics.json'
        if diagpath.exists():
            diag = read(diagpath)
        else:
            diag = diagnostics(X, y, groups, read(split / 'development_records.json'), cfg, nested['balanced_accuracy'])
            write(diagpath, diag)
        choice = {'candidate': selected, 'comparison': comparison, 'selection_metric': 'mean_inner_grouped_balanced_accuracy',
                  'threshold_rule': 'native estimator decision; no threshold search', 'test_labels_loaded': False}
        if (out / 'selection.json').exists():
            require(read(out / 'selection.json') == choice, 'Frozen selection differs')
        else:
            write(out / 'selection.json', choice)
        model = estimator(selected, e['seed'])
        seconds, ws = fit(model, X, y, cfg['per_fit_seconds'])
        bundle = {'task_id': TASK, 'label_definition_version': LABEL_VERSION, 'extractor': contract(),
                  'feature_columns': FEATURE_COLUMNS, 'estimator': model, 'candidate': selected,
                  'train_ids': frame.recording_id.tolist(), 'train_groups': sorted(set(groups)),
                  'split_hash': sm['split_hash'], 'dataset_hash': sm['dataset_hash'], 'chart_cutoff': sm['chart_cutoff'],
                  'synthetic': sm['synthetic'], 'pinned_env': environment()}
        bundle['cohort'] = cfg.get('cohort', 'any_weekly_hit_vs_noncharted')
        joblib.dump(bundle, out / 'model.joblib')
        final = {'synthetic': sm['synthetic'], 'evaluation_status': 'synthetic_test_only' if sm['synthetic'] else 'nested_development',
                 'skipped_candidates': [c['id'] for c in cfg['candidates'] if c not in cs],
                 'nested_development': nested, 'ci95': cluster_interval(y, oof, groups, e['seed']), 'folds': outer_records,
                 'selected_candidate': selected, 'final_fit_seconds': seconds, 'final_fit_warnings': ws,
                 'test_accessed': False, 'review_flags': diag['review_flags']}
        if not (out / 'summary.json').exists():
            write(out / 'summary.json', final)
        write(out / 'config.json', cfg) if not (out / 'config.json').exists() else None
        # Do not include ephemeral writer lock in the immutable manifest.
        seal(out, {'kind': 'model', 'synthetic': sm['synthetic'], 'label_definition_version': LABEL_VERSION,
                   'extractor': contract(), 'feature_schema_hash': digest(FEATURE_COLUMNS), 'group_policy_version': GROUP_VERSION,
                   'split_hash': sm['split_hash'], 'split_manifest_hash': sha(split / 'manifest.json'),
                   'dataset_hash': sm['dataset_hash'], 'chart_cutoff': sm['chart_cutoff'], 'code': code_identity(),
                   'pinned_env': environment(), 'config_hash': digest(cfg), 'evaluation_status': final['evaluation_status'], 'train_ids': bundle['train_ids'],
                   'train_groups': bundle['train_groups'], 'candidate': selected, 'cohort': bundle['cohort']})
        return final
    finally:
        if lock.exists():
            lock.unlink()


def main():
    p = parser(__doc__); p.add_argument('--split', required=True); p.add_argument('--out', required=True); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); print(train(a.split, a.out, a.resume, a.dry_run))


if __name__ == '__main__':
    cli(main)
