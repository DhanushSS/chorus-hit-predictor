"""Freeze one artist-connected holdout; publish development and sealed test files separately."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from chorus_hit.config import FEATURE_COLUMNS
from . import GROUP_VERSION
from .audit import assert_isolated, connected_groups
from .build_dataset import load_dataset, safe_output
from .common import cli, digest, parser, read, seal, sha, stage, verify, write
from .schema import require


def grouped_folds(y, groups, n, seed):
    y, groups = np.asarray(y), np.asarray(groups)
    require(set(y) == {0, 1}, 'Both classes required')
    for label in [0, 1]:
        require(len(set(groups[y == label])) >= n, f'Insufficient independent groups for {n} folds, class {label}')
    result = list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    for train, valid in result:
        require(set(y[train]) == {0, 1} and set(y[valid]) == {0, 1}, 'Fixed fold lacks a class; stop, do not search seeds')
        require(not set(groups[train]) & set(groups[valid]), 'Fold group overlap')
    return result


def freeze(dataset, destination, dry_run=False):
    X, rows, dm, cfg = load_dataset(dataset)
    c = cfg['evaluation']
    require(len(rows) > 0, 'No verified audio rows')
    if not dm['synthetic']:
        for label in [0, 1]:
            require(sum(r['label'] == label and r['chorus_annotation_status'] == 'reviewed' for r in rows) >= max(5, cfg['minimum_reviewed_per_class']), 'Real human chorus review gate incomplete')
    y = np.array([r['label'] for r in rows]); groups = np.array(connected_groups(rows))
    development, locked = next(GroupShuffleSplit(n_splits=1, test_size=c['test_fraction'], random_state=c['seed']).split(X, y, groups))
    require(set(y[development]) == {0, 1} and set(y[locked]) == {0, 1}, 'Fixed holdout lacks a class; no seed search')
    for part in [development, locked]:
        for label in [0, 1]:
            require(len(set(groups[part][y[part] == label])) >= (c['minimum_test_groups_per_class'] if dm['synthetic'] else max(5, c['minimum_test_groups_per_class'])), 'Insufficient independent class support')
    grouped_folds(y[development], groups[development], c['outer_folds'], c['seed'])
    for tr, _ in grouped_folds(y[development], groups[development], c['outer_folds'], c['seed']):
        grouped_folds(y[development][tr], groups[development][tr], c['inner_folds'], c['seed'])
    ids = [r['recording_id'] for r in rows]
    assert_isolated(rows, [ids[i] for i in development], [ids[i] for i in locked])
    detail = {'development': [ids[i] for i in development], 'locked': [ids[i] for i in locked],
              'group_by_id': dict(zip(ids, groups.tolist())), 'seed': c['seed'], 'group_policy_version': GROUP_VERSION,
              'rationale': 'One seeded GroupShuffleSplit on connected artist/work/album/match/audio components; no retries',
              'class_counts': {name: {str(label): int(sum(y[index] == label)) for label in [0, 1]} for name, index in [('development', development), ('locked', locked)]}}
    if dry_run:
        return detail
    out = safe_output(destination, dm['synthetic'])
    with stage(out) as tmp:
        for name, index in [('development', development), ('locked', locked)]:
            table = X.iloc[index].copy()
            table.insert(1, 'label', y[index]); table.insert(2, 'group', groups[index])
            table.to_csv(tmp / (name + '.csv'), index=False, float_format='%.17g')
        write(tmp / 'development_records.json', [rows[i] for i in development])
        # Test identity metadata is sufficient for leakage validation; no test labels/features here.
        identity_keys = ['recording_id', 'canonical_artist_ids', 'canonical_work_id', 'canonical_recording_id', 'album_id',
                         'match_set_id', 'matched_positive_ids', 'dedupe_group_id', 'audio_sha256', 'feature_row_hash',
                         'perceptual_duplicate_group', 'perceptual_signature']
        write(tmp / 'identity.json', [{k: r[k] for k in identity_keys if k in r} for r in rows])
        write(tmp / 'split.json', detail); write(tmp / 'config.json', cfg)
        seal(tmp, {'kind': 'split', 'synthetic': dm['synthetic'], 'dataset_hash': dm['dataset_hash'],
                   'dataset_manifest_hash': sha(Path(dataset) / 'manifest.json'), 'extractor': dm['extractor'],
                   'pinned_env': dm['pinned_env'], 'chart_cutoff': dm['chart_cutoff'], 'split_hash': digest(detail),
                   'config_hash': digest(cfg), 'group_policy_version': GROUP_VERSION, 'fresh_test_policy': 'one_access_per_split'})
    return detail


def load_development(folder):
    folder = Path(folder); m = verify(folder, 'split')
    detail = read(folder / 'split.json'); cfg = read(folder / 'config.json')
    require(digest(detail) == m['split_hash'] and digest(cfg) == m['config_hash'], 'Split/config hash mismatch')
    require(m['group_policy_version'] == GROUP_VERSION, 'Grouping policy mismatch')
    rows = read(folder / 'identity.json')
    ids = [r['recording_id'] for r in rows]
    require(len(ids) == len(set(ids)) and set(ids) == set(detail['development']) | set(detail['locked']), 'Split identity membership mismatch')
    require(len(detail['development']) + len(detail['locked']) == len(ids), 'Split duplicated/missing membership')
    require(all(type(c) is int and c >= 1 for c in cfg['evaluation'].values() if not isinstance(c, float)), 'Invalid split configuration')
    recomputed = dict(zip(ids, connected_groups(rows)))
    require(recomputed == detail['group_by_id'], 'Group identities differ from connected evidence')
    assert_isolated(rows, detail['development'], detail['locked'])
    X = pd.read_csv(folder / 'development.csv', float_precision='round_trip')
    require(list(X.columns) == ['recording_id', 'label', 'group'] + FEATURE_COLUMNS, 'Development schema mismatch')
    require(X.recording_id.tolist() == detail['development'], 'Development membership mismatch')
    require(X.group.tolist() == [recomputed[x] for x in X.recording_id], 'Development group mismatch')
    require(set(X.label) == {0, 1} and np.isfinite(X[FEATURE_COLUMNS].to_numpy()).all(), 'Invalid development values')
    records = read(folder / 'development_records.json')
    require([r['recording_id'] for r in records] == X.recording_id.tolist() and [r['label'] for r in records] == X.label.tolist(), 'Development label metadata mismatch')
    for r, values in zip(records, X[FEATURE_COLUMNS].to_numpy()):
        require(digest(values.tolist()) == r['feature_row_hash'], 'Development feature hash mismatch')
    return X, m, cfg


def main():
    p = parser(__doc__); p.add_argument('--dataset', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); print(freeze(a.dataset, a.out, a.dry_run))


if __name__ == '__main__':
    cli(main)
