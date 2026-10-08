"""Task-specific trusted-local bundle validation; never reinterpret a legacy model."""
from pathlib import Path
import joblib
import numpy as np
from chorus_hit.config import FEATURE_COLUMNS
from . import TASK, LABEL_VERSION, GROUP_VERSION, LABELS
from .audio import contract
from .common import digest, environment, read, sha, verify
from .modeling import estimator, score_values
from .schema import require, protocol


def parameter_signature(model):
    return {k: repr(v) for k, v in model.get_params(deep=True).items()}


def load(folder, allow_synthetic=False, require_evaluated=False):
    folder = Path(folder); m = verify(folder, 'model')
    require(m['synthetic'] is False or allow_synthetic, 'Synthetic fixture cannot serve real predictions')
    require(m['pinned_env'] == environment() and m['extractor'] == contract(), 'Model environment/extractor mismatch')
    require(m['feature_schema_hash'] == digest(FEATURE_COLUMNS) and m['label_definition_version'] == LABEL_VERSION, 'Model feature/label schema mismatch')
    require(m['group_policy_version'] == GROUP_VERSION, 'Wrong grouping policy')
    cfg = read(folder / 'config.json'); protocol(cfg)
    require(m['chart_cutoff'] == cfg['chart_cutoff'] and m['config_hash'] == digest(cfg), 'Model cutoff/config mismatch')
    # joblib is pickle: only locally generated, hash-checked, trusted artifacts.
    b = joblib.load(folder / 'model.joblib')
    for k in ['task_id', 'label_definition_version', 'extractor', 'pinned_env', 'candidate', 'dataset_hash', 'split_hash', 'chart_cutoff', 'synthetic', 'train_ids', 'train_groups']:
        require(b[k] == m[k], f'Bundle/manifest mismatch: {k}')
    require(b['task_id'] == TASK and b['feature_columns'] == FEATURE_COLUMNS, 'Cross-task feature bundle')
    require(len(b['train_ids']) == len(set(b['train_ids'])) and bool(b['train_ids']), 'Invalid model training membership')
    require(parameter_signature(b['estimator']) == parameter_signature(estimator(b['candidate'], cfg['evaluation']['seed'])), 'Estimator configuration mismatch')
    require(b['estimator'].feature_names_in_.tolist() == FEATURE_COLUMNS, 'Fitted feature order mismatch')
    require(b['estimator'].classes_.tolist() == [0, 1], 'Wrong fitted classes')
    if require_evaluated:
        e = read(folder / 'locked_evaluation.json')
        require(e['model_manifest_hash'] == sha(folder / 'manifest.json') and e['synthetic'] is False and e['evaluation_status'] == 'locked_test_once', 'No compatible genuine locked evaluation')
    return b, m


def predict_features(bundle, X, extractor):
    require(bundle['task_id'] == TASK and bundle['extractor'] == extractor == contract(), 'Wrong inference task/extractor')
    require(list(X.columns) == FEATURE_COLUMNS and len(X) > 0, 'Exact audio-only feature order required')
    require(np.isfinite(X.to_numpy()).all(), 'Nonfinite prediction features')
    predicted = bundle['estimator'].predict(X)
    scores, semantics = score_values(bundle['estimator'], X)
    return {'task_id': TASK, 'predictions': predicted.tolist(), 'labels': [LABELS[int(p)] for p in predicted],
            'scores': scores.tolist(), 'score_semantics': {**semantics, 'calibrated': False},
            'chart_cutoff': bundle['chart_cutoff'], 'use': 'Retrospective chart-membership classification; not future success probability'}
