"""One recorded holdout access after immutable development selection, shared by every run."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from chorus_hit.config import FEATURE_COLUMNS
from .artifacts import load, predict_features
from .audio import contract
from .common import cli, parser, read, sha, stamp, write
from .freeze_split import load_development
from .modeling import cluster_interval, metrics
from .schema import require


def evaluate(split, run, dry_run=False, allow_synthetic=False):
    split, run = Path(split), Path(run)
    dev, sm, cfg = load_development(split)
    b, m = load(run, allow_synthetic)
    require(m['split_hash'] == sm['split_hash'] and m['split_manifest_hash'] == sha(split / 'manifest.json'), 'Wrong locked split')
    require(m['dataset_hash'] == sm['dataset_hash'] and m['synthetic'] is sm['synthetic'], 'Wrong dataset')
    require(b['train_ids'] == dev.recording_id.tolist() and b['train_groups'] == sorted(set(dev.group)), 'Model not fit on frozen development membership')
    flags = read(run / 'diagnostics.json')['review_flags']
    require(not flags or sm['synthetic'], 'Development leakage/confounding review flags unresolved: ' + ', '.join(flags))
    require(not (split / 'test_access.json').exists(), 'Locked test already accessed; no second evaluation permitted')
    require(not (run / 'locked_evaluation.json').exists(), 'Run already evaluated')
    record = {'model_manifest_hash': sha(run / 'manifest.json'), 'split_hash': sm['split_hash'], 'access_started_at': stamp(),
              'policy': 'One access; an interrupted evaluation consumes the holdout too'}
    if dry_run:
        return {**record, 'test_labels_loaded': False, 'will_write': False}
    write(split / 'test_access.json', record)  # exclusive BEFORE reading any test labels
    table = pd.read_csv(split / 'locked.csv', float_precision='round_trip')
    detail = read(split / 'split.json')
    require(list(table.columns) == ['recording_id', 'label', 'group'] + FEATURE_COLUMNS, 'Locked schema mismatch')
    require(table.recording_id.tolist() == detail['locked'] and table.group.tolist() == [detail['group_by_id'][x] for x in table.recording_id], 'Locked membership mismatch')
    require(set(table.label) == {0, 1}, 'Locked class support invalid')
    began = time.perf_counter(); predicted = predict_features(b, table[FEATURE_COLUMNS], contract())
    elapsed = time.perf_counter() - began
    y = table.label.to_numpy(); p = np.asarray(predicted['predictions'])
    report = {**record, 'synthetic': sm['synthetic'], 'evaluation_status': 'synthetic_test_only' if sm['synthetic'] else 'locked_test_once',
              'metrics': metrics(y, p, np.asarray(predicted['scores'])), 'ci95': cluster_interval(y, p, table.group.to_numpy(), cfg['evaluation']['seed']),
              'prediction_seconds': elapsed, 'feature_extraction_seconds': 'Recorded separately per source row; excluded from prediction timing',
              'predictions': [{'recording_id': rid, 'actual': int(actual), 'predicted': int(pred)} for rid, actual, pred in zip(table.recording_id, y, p)],
              'per_group': {g: {'rows': int(sum(table.group == g)), 'accuracy': float(np.mean(y[table.group == g] == p[table.group == g]))} for g in sorted(set(table.group))},
              'no_future_success_claim': True}
    write(run / 'locked_evaluation.json', report)
    return report


def main():
    p = parser(__doc__); p.add_argument('--split', required=True); p.add_argument('--run', required=True)
    p.add_argument('--synthetic-test-only', action='store_true')
    a = p.parse_args(); print(evaluate(a.split, a.run, a.dry_run, a.synthetic_test_only))


if __name__ == '__main__':
    cli(main)
