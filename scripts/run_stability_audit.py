"""Ten fixed grouped partitions of a frozen candidate; descriptive, not fresh."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import sys
import time
import traceback
import warnings
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from chorus_hit.audit import sha256_file, atomic_json
from chorus_hit.config import FEATURE_COLUMNS
from chorus_hit.data import load_data, normalize_artist
from chorus_hit.estimators import make_estimator
from chorus_hit.evaluation import grouped_folds, classification_metrics


def main():
    started = time.monotonic()
    config_path = ROOT / 'configs/stability_audit.json'
    c = json.loads(config_path.read_text())
    out = ROOT / 'results' / c['run_id']
    out.mkdir(exist_ok=False)
    state = {'status': 'running', 'config': c, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'trials': [],
        'source_hashes': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                          [config_path, Path(__file__), ROOT/c['data'], ROOT/c['split'],
                           *sorted((ROOT/'chorus_hit').glob('*.py'))]}}
    atomic_json(out / 'manifest.json', state)
    try:
        if sha256_file(ROOT/c['data']) != c['data_sha256']:
            raise ValueError('Unexpected data identity')
        data = load_data(ROOT/c['data'])
        split = pd.read_csv(ROOT/c['split'])
        ids = set(split.loc[split.partition == c['partition'], 'track_id'])
        frame = data[data.track_id.isin(ids)].sort_values('track_id').reset_index(drop=True)
        if set(frame.track_id) != ids:
            raise ValueError('Development membership differs')
        groups = frame.artist.map(normalize_artist).to_numpy()
        y = frame.label.to_numpy()
        rows, folds, all_predictions = [], [], []
        for seed in c['seed_list']:
            per_seed = frame[['track_id', 'label']].copy()
            per_seed['artist_group'] = groups
            per_seed['seed'] = seed
            per_seed['fold'] = -1
            per_seed['lr_mi_50'] = -1
            per_seed['dummy'] = -1
            for fold, (fit, valid) in enumerate(grouped_folds(frame, c['folds'], seed, groups, FEATURE_COLUMNS)):
                per_seed.loc[valid, 'fold'] = fold
                folds.append({'seed': seed, 'fold': fold, 'fit_track_ids': frame.track_id.iloc[fit].tolist(),
                              'validation_track_ids': frame.track_id.iloc[valid].tolist()})
                for name in ['lr_mi_50', 'dummy']:
                    tick = time.monotonic()
                    estimator = make_estimator(c['fixed_model'], FEATURE_COLUMNS, c['estimator_seed']) if name == 'lr_mi_50' else DummyClassifier(strategy='prior')
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        estimator.fit(frame.iloc[fit][FEATURE_COLUMNS], y[fit])
                        prediction = estimator.predict(frame.iloc[valid][FEATURE_COLUMNS])
                        train_prediction = estimator.predict(frame.iloc[fit][FEATURE_COLUMNS])
                    model_path = out / f'{name}_seed{seed}_fold{fold}.joblib'
                    joblib.dump(estimator, model_path)
                    if not np.array_equal(joblib.load(model_path).predict(frame.iloc[valid][FEATURE_COLUMNS]), prediction):
                        raise ValueError('Reload predictions differ')
                    per_seed.loc[valid, name] = prediction
                    metrics = classification_metrics(y[valid], prediction)
                    state['trials'].append({'seed': seed, 'fold': fold, 'model': name, 'status': 'complete',
                        'fit_rows': len(fit), 'validation_rows': len(valid), 'metrics': metrics,
                        'training_balanced_accuracy': classification_metrics(y[fit], train_prediction)['balanced_accuracy'],
                        'seconds': time.monotonic()-tick, 'warnings': [str(w.message) for w in caught]})
                    atomic_json(out / 'manifest.json', state)
            if (per_seed[['fold','lr_mi_50','dummy']] < 0).any().any():
                raise ValueError('Missing out-of-fold prediction')
            for name in ['lr_mi_50', 'dummy']:
                metrics = classification_metrics(y, per_seed[name].to_numpy())
                trials = [t for t in state['trials'] if t['seed']==seed and t['model']==name]
                rows.append({'seed': seed, 'model': name, **{k: metrics[k] for k in ['accuracy','balanced_accuracy','precision','recall','f1']},
                    'fold_mean_balanced_accuracy': float(np.mean([t['metrics']['balanced_accuracy'] for t in trials])),
                    'fold_sd_balanced_accuracy': float(np.std([t['metrics']['balanced_accuracy'] for t in trials], ddof=1))})
            all_predictions.append(per_seed)
            print(json.dumps(rows[-2:]), flush=True)
        pd.concat(all_predictions).to_csv(out/'predictions.csv', index=False)
        table = pd.DataFrame(rows)
        table.to_csv(out/'repeat_metrics.csv', index=False)
        atomic_json(out/'folds.json', folds)
        summary = {'rows': len(frame), 'artist_groups': len(np.unique(groups)), 'repeats': len(c['seed_list']),
            'interpretation': c['purpose'], 'repeat_level_confidence_interval': None, 'model_promoted': False, 'models': {}}
        for name in ['lr_mi_50','dummy']:
            a = table[table.model==name]
            summary['models'][name] = {k: {'mean':float(a[k].mean()), 'sd_between_seeds':float(a[k].std()),
                'min':float(a[k].min()), 'max':float(a[k].max())} for k in ['accuracy','balanced_accuracy','precision','recall','f1','fold_mean_balanced_accuracy']}
        pivot = table.pivot(index='seed',columns='model',values='balanced_accuracy')
        summary['candidate_beats_dummy_repeats'] = int((pivot.lr_mi_50 > pivot.dummy).sum())
        summary['all_five_metrics_above_75_repeats'] = int((table[table.model=='lr_mi_50'][['accuracy','balanced_accuracy','precision','recall','f1']] > .75).all(axis=1).sum())
        atomic_json(out/'summary.json', summary)
        state.update(status='complete', seconds=time.monotonic()-started,
                     files={str(p.relative_to(out)):sha256_file(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
        atomic_json(out/'manifest.json', state)
        print(json.dumps(summary,indent=2),flush=True)
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        atomic_json(out/'manifest.json', state)
        raise


if __name__ == '__main__':
    main()
