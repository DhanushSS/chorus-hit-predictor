"""One-use evaluation of the locked HSP-S partition after development gates pass."""
from pathlib import Path
import joblib
import pandas as pd
from chorus_hit.hit_nonhit.common import digest, environment, read, sha, stamp, write
from chorus_hit.hit_nonhit.modeling import cluster_interval, score_values
from chorus_hit.hit_nonhit.schema import require
from . import TASK
from .data import verify
from .train import implementation
from .modeling import metrics


def load_model(run, require_evaluated=True):
    run = Path(run); manifest = verify(run, 'model')
    require(manifest['environment'] == environment(), 'Model environment differs')
    require(manifest['implementation'] == implementation(), 'Model implementation differs')
    if require_evaluated:
        result = read(run / 'locked_evaluation.json')
        require(result['model_manifest_sha256'] == sha(run / 'manifest.json'), 'Evaluation belongs to another model')
        require(result['split_manifest_sha256'] == manifest['split_manifest_sha256'], 'Evaluation split differs')
    bundle = joblib.load(run / 'model.joblib')  # trusted local artifact after manifest checks, never uploaded pickle
    require(bundle['task_id'] == TASK and digest(bundle['columns']) == bundle['feature_schema_hash'], 'Model task/schema differs')
    require(bundle['environment'] == manifest['environment'], 'Bundle environment differs')
    return bundle


def evaluate(split, run):
    split, run = Path(split), Path(run)
    sm = verify(split, 'split'); mm = verify(run, 'model')
    require(mm['split_manifest_sha256'] == sha(split / 'manifest.json'), 'Wrong split for model')
    require(not read(run / 'diagnostics.json')['review_flags'], 'Development confounding flags block locked evaluation')
    require(not (split / 'test_access.json').exists(), 'Locked test already consumed')
    require(not (run / 'locked_evaluation.json').exists(), 'Locked result already exists')
    bundle = load_model(run, require_evaluated=False); cfg = read(split / 'config.json')
    # Exclusive publication happens before loading held-out features or labels.
    write(split / 'test_access.json', {'used_at': stamp(), 'model_manifest_sha256': sha(run / 'manifest.json'),
                                      'split_manifest_sha256': sha(split / 'manifest.json'), 'policy': 'one access; failure does not refund access'})
    frame = pd.read_parquet(split / 'locked.parquet'); detail = read(split / 'split.json')
    require(frame.uuid.tolist() == detail['locked']['ids'], 'Locked identities changed')
    X = frame[bundle['columns']]; y = frame.label.to_numpy(); model = bundle['model']
    predicted = model.predict(X); scores, semantics = score_values(model, X)
    result = {'status': 'TRAINED_AND_LOCKED_TESTED', 'task_id': TASK, 'evaluated_at': stamp(),
              'model_manifest_sha256': sha(run / 'manifest.json'), 'split_manifest_sha256': sha(split / 'manifest.json'),
              'metrics': metrics(y, predicted, scores), 'group_interval': cluster_interval(y, predicted, frame.group.to_numpy(), cfg['evaluation']['seed']),
              'score_semantics': semantics, 'rows': len(frame), 'groups': frame.group.nunique(),
              'uuid': frame.uuid.tolist(), 'label': y.tolist(), 'predicted': predicted.tolist(), 'scores': scores.tolist(),
              'scope': 'One-use retrospective HSP-S publisher-label benchmark; not repeated-chorus or future success prediction'}
    write(run / 'locked_evaluation.json', result)
    return {k: v for k, v in result.items() if k not in {'uuid', 'label', 'predicted', 'scores'}}


def predict_frame(run, frame):
    bundle = load_model(run)
    require(list(frame.columns) == bundle['columns'], 'Upload requires the exact ordered HSP-S scalar audio columns')
    require(0 < len(frame) <= 100, 'Supply 1 to 100 feature records')
    numeric = frame.apply(pd.to_numeric, errors='raise')
    import numpy as np
    require(not np.isinf(numeric.to_numpy()).any(), 'Infinite feature values rejected')
    require(numeric.notna().any(axis=1).all(), 'Empty feature records rejected')
    scores, semantics = score_values(bundle['model'], numeric)
    return {'labels': [bundle['labels'][str(int(x))] for x in bundle['model'].predict(numeric)], 'scores': scores.tolist(),
            'score_semantics': semantics, 'task_id': TASK}
