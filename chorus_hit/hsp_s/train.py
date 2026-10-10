"""Resumable nested group validation on HSP-S development data only."""
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from chorus_hit.config import ROOT
from chorus_hit.hit_nonhit.common import digest, environment, read, runtime_check, sha, write
from chorus_hit.hit_nonhit.freeze_split import grouped_folds
from chorus_hit.hit_nonhit.modeling import cluster_interval, score_values
from chorus_hit.hit_nonhit.schema import require
from chorus_hit.hit_nonhit.train import fit
from .data import development, seal
from .modeling import estimator, metrics


def implementation():
    paths = list((ROOT / 'chorus_hit/hsp_s').glob('*.py')) + [ROOT / ('chorus_hit/hit_nonhit/' + n) for n in
            ['train.py', 'modeling.py', 'fit_worker.py', 'audit.py', 'freeze_split.py', 'common.py']]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def diagnostics(frame, columns, cfg, reference):
    y = frame.label.to_numpy(); groups = frame.group.to_numpy(); e = cfg['evaluation']
    folds = grouped_folds(y, groups, e['inner_folds'], e['seed'])
    categorical = ['nuisance_codec', 'nuisance_extractor']; numeric = ['nuisance_sample_rate', 'nuisance_duration']
    predicted = np.empty(len(y), dtype=int)
    for tr, va in folds:
        model = Pipeline([('input', ColumnTransformer([
            ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
            ('numeric', Pipeline([('missing', SimpleImputer()), ('scale', StandardScaler())]), numeric)])),
            ('model', LogisticRegression(C=.1, class_weight='balanced', max_iter=2000))])
        fit(model, frame.iloc[tr][categorical + numeric], y[tr], cfg['per_fit_seconds'])
        predicted[va] = model.predict(frame.iloc[va][categorical + numeric])
    nuisance = metrics(y, predicted)
    rng = np.random.default_rng(e['seed']); scores = []
    mixed = [g for g in np.unique(groups) if len(set(y[groups == g])) == 2]
    candidate = next(c for c in cfg['candidates'] if c['family'] == 'logistic')
    for _ in range(cfg['permutation_repeats'] if mixed else 0):
        shuffled = y.copy()
        for g in mixed:
            ix = np.flatnonzero(groups == g); shuffled[ix] = rng.permutation(y[ix])
        p = np.empty(len(y), dtype=int)
        for tr, va in folds:
            model = estimator(candidate, columns, e['seed'])
            fit(model, frame.iloc[tr][columns], shuffled[tr], cfg['per_fit_seconds'])
            p[va] = model.predict(frame.iloc[va][columns])
        scores.append(metrics(shuffled, p)['balanced_accuracy'])
    flags = []
    if nuisance['balanced_accuracy'] >= cfg['nuisance_block_threshold']: flags.append('source_codec_duration_only_balanced_accuracy_at_least_0.65')
    if not mixed: flags.append('within_group_permutation_not_identifiable')
    if scores and np.mean(scores) >= max(.6, reference - .02): flags.append('high_within_group_permuted_label_performance')
    return {'nuisance_only': nuisance, 'within_group_permutation_balanced_accuracy': scores,
            'exchangeable_groups': len(mixed), 'exchangeable_rows': int(np.isin(groups, mixed).sum()),
            'review_flags': flags, 'scope': 'Development only. Flags block locked evaluation; no automatic cohort or seed revision.'}


def train(split, destination, resume=False):
    runtime_check(); split = Path(split); out = Path(destination).resolve()
    require(out.is_relative_to(ROOT / 'results/hsp_s'), 'Use ignored results/hsp_s namespace')
    require(not (split / 'test_access.json').exists(), 'Final test already accessed; further selection prohibited')
    frame, columns, cfg, sm = development(split)
    require(not (out / 'manifest.json').exists(), 'Completed HSP-S run cannot be overwritten')
    require(resume or not out.exists(), 'Existing run requires explicit resume')
    out.mkdir(parents=True, exist_ok=True)
    identity = {'split_manifest_sha256': sha(split / 'manifest.json'), 'config_hash': digest(cfg), 'code': implementation(), 'environment': environment()}
    if (out / 'resume_identity.json').exists(): require(read(out / 'resume_identity.json') == identity, 'Resume identity changed')
    else: write(out / 'resume_identity.json', identity)
    with (out / '.training.lock').open('x'): pass
    try:
        X = frame[columns]; y = frame.label.to_numpy(); groups = frame.group.to_numpy(); e = cfg['evaluation']
        candidates = cfg['candidates']; outer = grouped_folds(y, groups, e['outer_folds'], e['seed'])
        require(len(candidates) == 27 and any(c['family'] == 'dummy' for c in candidates), 'Preregistered 27 candidates including dummy required')
        trials = out / 'trials'; trials.mkdir(exist_ok=True)
        def trial(c, tr, va, scope):
            key = digest([c, frame.uuid.iloc[tr].tolist(), frame.uuid.iloc[va].tolist(), scope])
            path = trials / (key + '.json')
            if path.exists():
                r = read(path); require(r['hash'] == digest({k: v for k, v in r.items() if k != 'hash'}), 'Trial checkpoint changed'); return r
            require(sum(read(p)['seconds'] for p in trials.glob('*.json')) < cfg['total_fit_budget_seconds'], 'Fit budget exhausted')
            r = {'candidate': c['id'], 'scope': scope, 'status': 'failed'}; start = time.monotonic()
            try:
                model = estimator(c, columns, e['seed']); _, warnings = fit(model, X.iloc[tr], y[tr], cfg['per_fit_seconds'])
                r.update(status='complete', validation=metrics(y[va], model.predict(X.iloc[va])), warnings=warnings)
            except (ValueError, RuntimeError, TimeoutError) as error: r['error'] = str(error)
            r['seconds'] = time.monotonic() - start; r['hash'] = digest(r); write(path, r)
            print(f"{scope}: {c['id']} {r['status']}", flush=True)
            return r
        def select(index, scope):
            folds = grouped_folds(y[index], groups[index], e['inner_folds'], e['seed']); scores = []
            for c in candidates:
                records = [trial(c, index[tr], index[va], f'{scope}/inner_{i}') for i, (tr, va) in enumerate(folds)]
                good = all(r['status'] == 'complete' for r in records)
                scores.append({'candidate_id': c['id'], 'balanced_accuracy': float(np.mean([r['validation']['balanced_accuracy'] for r in records])) if good else None,
                               'folds': [r.get('validation') for r in records], 'seconds': sum(r['seconds'] for r in records),
                               'failed_folds': sum(r['status'] != 'complete' for r in records)})
            valid = [r for r in scores if r['balanced_accuracy'] is not None]; require(valid, 'All models failed')
            winner = sorted(valid, key=lambda r: (-r['balanced_accuracy'], r['candidate_id']))[0]['candidate_id']
            return next(c for c in candidates if c['id'] == winner), scores
        oof = np.empty(len(y), dtype=int); outer_results = []
        for fold, (tr, va) in enumerate(outer):
            c, comparison = select(tr, f'outer_{fold}'); path = out / f'outer_{fold}.json'
            if path.exists():
                r = read(path); require(r['hash'] == digest({k: v for k, v in r.items() if k != 'hash'}), 'Outer checkpoint changed')
                require(r['candidate'] == c and r['validation_ids'] == frame.uuid.iloc[va].tolist(), 'Outer identity changed')
            else:
                model = estimator(c, columns, e['seed']); seconds, warnings = fit(model, X.iloc[tr], y[tr], cfg['per_fit_seconds'])
                r = {'fold': fold, 'candidate': c, 'comparison': comparison, 'validation_ids': frame.uuid.iloc[va].tolist(),
                     'predicted': model.predict(X.iloc[va]).tolist(), 'seconds': seconds, 'warnings': warnings}
                r['hash'] = digest(r); write(path, r)
            oof[va] = r['predicted']; outer_results.append(r)
            print(f'Outer fold {fold + 1}/{len(outer)} complete', flush=True)
        chosen, comparison = select(np.arange(len(y)), 'final_development')
        nested = metrics(y, oof)
        if (out / 'diagnostics.json').exists():
            diag = read(out / 'diagnostics.json')
            require(diag['payload_sha256'] == digest({k: v for k, v in diag.items() if k != 'payload_sha256'}), 'Diagnostic checkpoint changed')
        else:
            diag = diagnostics(frame, columns, cfg, nested['balanced_accuracy']); diag['payload_sha256'] = digest(diag); write(out / 'diagnostics.json', diag)
        model = estimator(chosen, columns, e['seed']); seconds, warnings = fit(model, X, y, cfg['per_fit_seconds'])
        train_scores, semantics = score_values(model, X)
        bundle = {'task_id': cfg['task_id'], 'model': model, 'columns': columns, 'feature_schema_hash': digest(columns),
                  'split_manifest_sha256': identity['split_manifest_sha256'], 'environment': environment(), 'score_semantics': semantics,
                  'labels': {'0': 'No chart entry in HSP-S benchmark', '1': 'Billboard chart entry in HSP-S benchmark'}}
        joblib.dump(bundle, out / 'model.joblib', compress=3)
        summary = {'status': 'TRAINED_DEV_ONLY', 'task_id': cfg['task_id'], 'development_rows': len(y), 'development_groups': len(set(groups)),
                   'selected_candidate': chosen, 'nested_development': nested, 'nested_group_interval': cluster_interval(y, oof, groups, e['seed']),
                   'resubstitution_training': metrics(y, model.predict(X), train_scores), 'final_fit_seconds': seconds, 'warnings': warnings,
                   'candidate_comparison': comparison, 'locked_test_access': 'NOT_USED', 'review_flags': diag['review_flags'],
                   'feature_count': len(columns), 'source': 'HSP-S AcousticBrainz CC BY 4.0', 'audio_or_chorus_extraction': False}
        write(out / 'summary.json', summary); write(out / 'config.json', cfg)
        write(out / 'development_predictions.json', {'uuid': frame.uuid.tolist(), 'label': y.tolist(), 'group': groups.tolist(), 'predicted': oof.tolist()})
        (out / '.training.lock').unlink()
        seal(out, 'model', {'split_manifest_sha256': identity['split_manifest_sha256'], 'environment': environment(), 'implementation': identity['code']})
        return summary
    finally:
        (out / '.training.lock').unlink(missing_ok=True)
