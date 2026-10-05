"""One fixed ensemble experiment on the original development partition only."""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from chorus_hit.artifacts import load_run
from chorus_hit.audit import atomic_json, sha256_file
from chorus_hit.config import ROOT
from chorus_hit.contracts import frozen_split
from chorus_hit.data import normalize_artist
from chorus_hit.diagnostics import bounded_command, identity, new_output
from chorus_hit.estimators import make_estimator
from chorus_hit.evaluation import bootstrap_inputs, classification_metrics, grouped_folds

CONTROL = 'v4_lr_std74_standard_0.1'
MEMBERS = tuple(f'v4_lr_std74_{scaling}_{c}'
                for scaling in ('standard', 'quantile') for c in (0.003, 0.01, 0.03, 0.1))
METRICS = ('accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1')


def declared_members(config):
    candidates = config['candidates']
    if len({s['id'] for s in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate IDs')
    by_id = {s['id']: s for s in candidates}
    selected = []
    for name in MEMBERS:
        spec = by_id[name]
        scaling, c = name.rsplit('_', 2)[-2:]
        representation = {'kind': 'none', 'columns': 'std_only', 'scaling': scaling}
        if scaling == 'quantile':
            representation['n_quantiles'] = 100
        if (spec['family'] != 'logistic'
                or spec['params'] != {'C': float(c), 'class_weight': 'balanced'}
                or spec['representation'] != representation):
            raise ValueError('Member settings differ from the declared final experiment')
        selected.append(spec)
    return selected


def development_input(config):
    run = load_run('v4_std74_001')
    split, development_ids, historical_ids = frozen_split(
        run.data, ROOT / config['split_manifest'], config['split_sha256'])
    if run.manifest['dataset']['source_sha256'] != config['data_sha256']:
        raise ValueError('Dataset/config hash mismatch')
    membership = split.set_index('track_id').loc[run.data.track_id]
    dev = run.data.loc[membership.partition.to_numpy() == 'train'].reset_index(drop=True)
    if set(dev.track_id) != set(development_ids) or set(dev.track_id) & set(historical_ids):
        raise ValueError('Development membership mismatch')
    return dev, run.bundle['features']


def average_predictions(probabilities):
    """Uncalibrated equal-weight probabilities; an exact 0.5 tie is class 0."""
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 2 or not all(p.shape) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()):
        raise ValueError('Expected a nonempty rows-by-members probability matrix in [0,1]')
    score = p.mean(axis=1)
    return (score > 0.5).astype(int), score


def collect_predictions(dev, features, specs, folds):
    probabilities = np.full((len(dev), len(specs)), np.nan)
    control = np.full(len(dev), -1, dtype=int)
    fold_ids = np.full(len(dev), -1, dtype=int)
    fit_records = []
    with threadpool_limits(limits=1):
        for fold, (fit, val) in enumerate(folds):
            if (fold_ids[val] >= 0).any() or set(fit) & set(val):
                raise ValueError('Fold rows overlap or repeat')
            fold_ids[val] = fold
            for member, spec in enumerate(specs):
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    model = make_estimator(spec, features, 42)
                    start = time.perf_counter()
                    model.fit(dev.iloc[fit][features], dev.label.iloc[fit])
                    elapsed = time.perf_counter() - start
                    if list(model.classes_) != [0, 1]:
                        raise ValueError('Expected ordered classes [0,1]')
                    p = np.asarray(model.predict_proba(dev.iloc[val][features]), dtype=float)
                    if (p.shape != (len(val), 2) or not np.isfinite(p).all()
                            or (p < 0).any() or (p > 1).any()
                            or not np.allclose(p.sum(axis=1), 1)):
                        raise ValueError('Invalid class probabilities')
                    probabilities[val, member] = p[:, 1]
                    if spec['id'] == CONTROL:
                        control[val] = model.predict(dev.iloc[val][features])
                fit_records.append({'fold': fold, 'member': spec['id'], 'fit_seconds': elapsed,
                                    'warnings': [str(w.message) for w in captured]})
            print(f'Completed fold {fold + 1}/{len(folds)}', flush=True)
    if (fold_ids < 0).any() or (control < 0).any():
        raise ValueError('Incomplete out-of-fold coverage or missing control')
    ensemble, scores = average_predictions(probabilities)
    rows = pd.DataFrame({'track_id': dev.track_id, 'artist_group': dev.artist.map(normalize_artist),
                         'label': dev.label, 'fold': fold_ids, 'control_prediction': control,
                         'ensemble_prediction': ensemble, 'ensemble_score': scores})
    for member, spec in enumerate(specs):
        rows[f"probability_{spec['id']}"] = probabilities[:, member]
    return rows, fit_records


def paired_differences(y, ensemble, control, groups, repeats=2000, seed=2026):
    y, (ensemble, control), groups, _ = bootstrap_inputs(y, [ensemble, control], groups, repeats)
    bins = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in METRICS}
    for _ in range(repeats):
        ix = np.concatenate([bins[g] for g in rng.integers(len(bins), size=len(bins))])
        if len(np.unique(y[ix])) != 2:
            continue
        m1, m0 = classification_metrics(y[ix], ensemble[ix]), classification_metrics(y[ix], control[ix])
        for key in METRICS:
            samples[key].append(m1[key] - m0[key])
    m1, m0 = classification_metrics(y, ensemble), classification_metrics(y, control)
    return {
        'metrics': {k: {'difference': m1[k] - m0[k],
                        'ci95': np.quantile(v, [.025, .975]).tolist() if v else None}
                    for k, v in samples.items()},
        'requested_resamples': repeats, 'valid_resamples': len(samples['accuracy']),
        'unavailable_reason': None if samples['accuracy'] else 'No two-class resamples',
        'method': 'Paired percentile bootstrap of whole normalized artist-name groups',
        'limitations': 'Conditional on fixed predictions; dependent folds and prior development reuse '
                      'are not accounted for. This is not independent confirmation.'}


def run(destination, *, _reserved=False):
    destination = Path(destination) if _reserved else new_output(destination)
    config_path = ROOT / 'configs/v4_std74.json'
    config = json.loads(config_path.read_text())
    specs = declared_members(config)
    dev, features = development_input(config)
    folds = grouped_folds(dev, 5, 44, dev.artist.map(normalize_artist), features)
    protocol = identity(
        __file__, status='exploratory_fixed_ensemble_development', fresh_test=False,
        hypothesis='Equal averaging may reduce variance across existing std74 logistic settings.',
        config_sha256=sha256_file(config_path), split_sha256=config['split_sha256'],
        members=specs, weights=[1 / len(specs)] * len(specs), control=CONTROL,
        folds=[{'fit': dev.track_id.iloc[f].tolist(), 'validation': dev.track_id.iloc[v].tolist()}
               for f, v in folds], fold_seed=44, estimator_seed=42, bootstrap_seed=2026,
        bootstrap_repeats=2000, fit_count=40, workers=1,
        score_kind='uncalibrated mean class-1 probability', prediction_rule='score > 0.5; tie -> 0',
        primary_metric='accuracy', secondary_metrics=list(METRICS[1:]),
        decision_rule='Report observed accuracy change against the fixed control. A positive change '
                      'with a paired accuracy CI above zero and nondecreasing balanced accuracy '
                      'is stronger exploratory evidence only; never auto-promote.',
        stopping_rule='One configuration, one five-fold run. Stop regardless of outcome; '
                      'no seed, threshold, weight, member or historical-set search.',
        previous_exposure='Earlier nested, fixed-pipeline, threshold and all-751-song diagnostics '
                          'already inspected these data. No fresh evaluation exists.',
        changes_to_active_model=False)
    # Persist the complete specification before the first estimator is fitted.
    atomic_json(destination / 'protocol.json', protocol)
    rows, fit_records = collect_predictions(dev, features, specs, folds)
    rows.to_csv(destination / 'predictions.csv', index=False)
    metrics = {name: classification_metrics(rows.label, rows[f'{name}_prediction'])
               for name in ('control', 'ensemble')}
    paired = paired_differences(rows.label, rows.ensemble_prediction, rows.control_prediction,
                                rows.artist_group)
    ci = paired['metrics']['accuracy']['ci95']
    result = {'status': 'exploratory_fixed_ensemble_development', 'fresh_test': False,
              'protocol_sha256': sha256_file(destination / 'protocol.json'),
              'predictions_sha256': sha256_file(destination / 'predictions.csv'),
              'n': len(dev), 'metrics': metrics, 'paired': paired, 'fits': fit_records,
              'observed_accuracy_improvement': metrics['ensemble']['accuracy'] > metrics['control']['accuracy'],
              'stronger_exploratory_evidence': bool(ci is not None and ci[0] > 0
                  and metrics['ensemble']['balanced_accuracy'] >= metrics['control']['balanced_accuracy']),
              'promotion': False, 'further_accuracy_experiments': 'stopped per user request'}
    atomic_json(destination / 'result.json', result)
    print(json.dumps({'metrics': metrics, 'paired': paired, 'promotion': False}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        run(args.out, _reserved=True)
    else:
        bounded_command([sys.executable, '-m', 'scripts.final_accuracy_check', '--out',
                         str(args.out.resolve()), '--_worker'], args.out, 120)


if __name__ == '__main__':
    main()
