"""Exact Essentia scalar mapping. Schema coverage is not scientific parity."""
import math
import numpy as np

PROFILE_ID = 'essentia-musicextractor-research-unvalidated-v1'
AUDIO_PREDICTION_VALIDATED = False


def map_features(pool, columns):
    missing = [key for key in columns if key not in pool]
    if missing: raise ValueError(f'Extractor coverage {len(columns)-len(missing)}/{len(columns)}; missing: {", ".join(missing[:8])}. No feature values will be invented.')
    result = []
    for key in columns:
        value = pool[key]
        if isinstance(value, (bool, str)) or not np.isscalar(value): raise ValueError(f'{key}: expected one numeric scalar, not a vector/string.')
        if not math.isfinite(float(value)): raise ValueError(f'{key}: non-finite extractor feature.')
        result.append(float(value))
    return result


def audio_prediction(*args, **kwargs):
    # Intentionally no user-supplied boolean/proof override. Enable only in a reviewed future release.
    raise ValueError('Audio prediction is not validated/available: matched-recording parity and independent audio evaluation are missing.')
