"""Fixed, development-only representation pilot; no verified recording claims."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import hashlib
import json
import re
import subprocess
import time
import traceback
import unicodedata
import warnings
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from chorus_hit.estimators import make_estimator
from chorus_hit.evaluation import classification_metrics


def norm(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value)).casefold()).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run():
    start = time.monotonic()
    config_path = ROOT / 'configs/hf_feature_pilot.json'
    config = json.loads(config_path.read_text())
    source = ROOT / 'results/hf_feasibility_001'
    out = ROOT / 'results' / config['run_id']
    out.mkdir(exist_ok=False)
    manifest = {'status': 'running', 'config': config, 'config_sha256': sha(config_path),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'script_sha256': sha(Path(__file__)), 'trials': []}
    save(out / 'manifest.json', manifest)
    try:
        raw = pd.read_csv(ROOT / 'data/chorus_features.csv')
        info = pd.read_csv(source / 'music4all_information.csv', sep='\t')
        folds = pd.read_csv(ROOT / 'results/v2/v3_nested_001/predictions.csv')
        features = raw.columns[4:].tolist()
        raw['key'] = raw.artist.map(norm) + '|' + raw.title.map(norm)
        info['key'] = info.artist.map(norm) + '|' + info.song.map(norm)
        raw_unique = raw[~raw.key.duplicated(keep=False)]
        info_unique = info[~info.key.duplicated(keep=False)]
        linked = raw_unique.merge(info_unique[['key', 'id', 'album_name']], on='key', validate='one_to_one')
        linked = linked.merge(folds[['track_id', 'artist_group', 'fold']], on='track_id', validate='one_to_one')
        manifest['development_title_matches'] = len(linked)
        external_path = source / 'onion_id_musicnn.tsv.bz2'
        assert hashlib.md5(external_path.read_bytes()).hexdigest() == config['source_md5']
        external = pd.read_csv(external_path, sep='\t')
        assert external.id.is_unique
        ext_columns = [c for c in external if c != 'id']
        external = external.rename(columns={c: 'musicnn_' + c for c in ext_columns})
        ext_columns = ['musicnn_' + c for c in ext_columns]
        linked = linked.merge(external, on='id', validate='one_to_one')
        assert np.isfinite(linked[ext_columns].to_numpy()).all()
        linked['feature_hash'] = [hashlib.sha256(row.tobytes()).hexdigest() for row in linked[ext_columns].to_numpy()]
        cross_group = linked.groupby('feature_hash').artist_group.nunique()
        excluded = linked[linked.feature_hash.isin(cross_group[cross_group > 1].index)]
        excluded[['track_id', 'id', 'artist_group', 'feature_hash']].to_csv(out / 'excluded_duplicates.csv', index=False)
        linked = linked[~linked.track_id.isin(excluded.track_id)].sort_values('track_id').reset_index(drop=True)
        assert len(linked) > 50 and linked.fold.nunique() == 5 and linked.label.nunique() == 2
        assert linked.groupby('artist_group').fold.nunique().max() == 1
        historical = set(pd.read_csv(ROOT / 'results/split_manifest.csv').query("partition == 'test'").track_id)
        assert not historical.intersection(linked.track_id)
        linked.to_csv(out / 'pilot_inputs.csv', index=False)
        linked[['track_id', 'id', 'artist', 'title', 'album_name', 'label', 'artist_group', 'fold']].to_csv(out / 'membership.csv', index=False)
        spec = {'family': 'logistic', 'params': {'C': config['logistic_C']}, 'representation': {'kind': 'mi', 'value': 50}}
        legacy = make_estimator(spec, features, seed=config['seed'])
        audio = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(C=config['logistic_C'], max_iter=config['max_iter'], random_state=config['seed']))])
        combined = Pipeline([('features', ColumnTransformer([
            ('legacy', clone(legacy[:-1]), features), ('musicnn', StandardScaler(), ext_columns)])),
            ('model', LogisticRegression(C=config['logistic_C'], max_iter=config['max_iter'], random_state=config['seed']))])
        arms = {'legacy_mi50': (legacy, features), 'musicnn': (audio, ext_columns), 'combined': (combined, features + ext_columns)}
        predictions = linked[['track_id', 'artist_group', 'label', 'fold']].copy()
        for arm in config['arms']:
            model, columns = arms[arm]
            predictions[arm] = -1
            predictions[arm + '_score'] = np.nan
            for fold in sorted(linked.fold.unique()):
                train = linked.fold != fold
                valid = ~train
                assert not set(linked.loc[train, 'artist_group']) & set(linked.loc[valid, 'artist_group'])
                assert linked.loc[train, 'label'].nunique() == linked.loc[valid, 'label'].nunique() == 2
                tick = time.monotonic()
                fitted = clone(model)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    fitted.fit(linked.loc[train, columns], linked.loc[train, 'label'])
                    pred = fitted.predict(linked.loc[valid, columns])
                    score = fitted.predict_proba(linked.loc[valid, columns])[:, 1]
                assert np.array_equal(pred, (score > config['threshold']).astype(int))
                artifact = out / f'{arm}_fold{fold}.joblib'
                joblib.dump(fitted, artifact)
                assert np.array_equal(joblib.load(artifact).predict(linked.loc[valid, columns]), pred)
                predictions.loc[valid, arm] = pred
                predictions.loc[valid, arm + '_score'] = score
                trial = {'arm': arm, 'fold': int(fold), 'fit_rows': int(train.sum()), 'validation_rows': int(valid.sum()),
                         'status': 'complete', 'seconds': time.monotonic() - tick, 'warnings': [str(w.message) for w in caught]}
                manifest['trials'].append(trial)
                save(out / 'manifest.json', manifest)
                print(json.dumps(trial), flush=True)
        assert predictions.notna().all().all()
        predictions.to_csv(out / 'predictions.csv', index=False)
        y = predictions.label.to_numpy()
        metrics = {arm: classification_metrics(y, predictions[arm].to_numpy()) for arm in arms}
        groups = predictions.artist_group.to_numpy()
        unique = np.unique(groups)
        rng = np.random.default_rng(config['seed'])
        boots = {arm: [] for arm in arms}
        for _ in range(config['bootstrap_repeats']):
            sampled = rng.choice(unique, len(unique), replace=True)
            ix = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
            if np.unique(y[ix]).size != 2:
                continue
            for arm in arms:
                correct = predictions[arm].to_numpy()[ix] == y[ix]
                boots[arm].append(float((correct[y[ix] == 0].mean() + correct[y[ix] == 1].mean()) / 2))
        for arm in arms:
            metrics[arm]['balanced_accuracy_ci95'] = np.quantile(boots[arm], [.025, .975]).tolist()
        comparisons = {arm: {'balanced_accuracy_difference': metrics[arm]['balanced_accuracy'] - metrics['legacy_mi50']['balanced_accuracy'],
                            'paired_ci95': np.quantile(np.asarray(boots[arm]) - boots['legacy_mi50'], [.025, .975]).tolist()}
                       for arm in ['musicnn', 'combined']}
        result = {'evaluation': 'exploratory_linkage_development_pilot', 'rows': len(linked), 'artist_groups': len(unique),
                  'class_counts': {str(k): int(v) for k, v in linked.label.value_counts().items()},
                  'excluded_cross_group_duplicates': len(excluded), 'external_dimensions': len(ext_columns),
                  'metrics': metrics, 'paired_vs_matched_legacy': comparisons, 'identity_verified': False,
                  'chorus_segment_equivalence_verified': False, 'fresh_test_available': False, 'model_promoted': False}
        save(out / 'summary.json', result)
        manifest.update(status='complete', seconds=time.monotonic() - start,
                        source_files={str(p.relative_to(ROOT)): sha(p) for p in [config_path, ROOT / 'data/chorus_features.csv', source / 'music4all_information.csv', external_path, ROOT / 'results/v2/v3_nested_001/predictions.csv']},
                        files={p.name: sha(p) for p in out.iterdir() if p.name != 'manifest.json'})
        save(out / 'manifest.json', manifest)
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc(), seconds=time.monotonic() - start)
        save(out / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    run()
