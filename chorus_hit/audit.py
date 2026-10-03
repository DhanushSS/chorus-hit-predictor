"""Explicit data identity, audit records and atomic JSON writes."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
import tempfile

import pandas as pd

from .config import DATA_PATH, FEATURE_COLUMNS

LEGACY_DATASET = 'chorus-chart-proxy-v1'
TASK_VERSION = 'year-end-v-other-chart-v1'
GROUP_VERSION = 'normalized-artist-string-v1'
LEGACY_SHA256 = '79951ffc393fc836405509326d1ee03efd96434327d800711c4b8ed817ef44ce'


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def frame_fingerprint(frame):
    # Includes ordered rows, values, columns and dtypes; independent of CSV byte formatting.
    h = hashlib.sha256(json.dumps([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)]).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


@dataclass
class DatasetContext:
    frame: pd.DataFrame
    source_path: Path
    source_sha256: str
    processed_fingerprint: str
    dataset_version: str
    features: list

    def verify(self):
        if sha256_file(self.source_path) != self.source_sha256:
            raise ValueError('Dataset source bytes changed after loading')
        if frame_fingerprint(self.frame) != self.processed_fingerprint:
            raise ValueError('Dataset frame was modified; create a versioned transformation')

    def identity(self):
        self.verify()
        return {'source_path': str(self.source_path), 'source_sha256': self.source_sha256,
                'processed_fingerprint': self.processed_fingerprint, 'dataset_version': self.dataset_version,
                'features': self.features, 'fingerprint_algorithm': 'pandas-hash-values+ordered-schema-v1'}


def load_dataset(path=DATA_PATH, *, features=None, dataset_version=LEGACY_DATASET, expected_sha256=None):
    from .data import load_data
    path = Path(path).resolve(); digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError('Dataset SHA-256 mismatch')
    features = list(FEATURE_COLUMNS if features is None else features)
    frame = load_data(path, features=features)
    return DatasetContext(frame, path, digest, frame_fingerprint(frame), dataset_version, features)


def main():
    from .data import data_audit, normalize_artist
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DATA_PATH); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset-version', default=LEGACY_DATASET)
    p.add_argument('--expected-sha256')
    args = p.parse_args()
    ctx = load_dataset(args.data, dataset_version=args.dataset_version, expected_sha256=args.expected_sha256)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out/'dataset_audit.json', {**data_audit(ctx.frame), **ctx.identity()})
    frame = ctx.frame
    frame.assign(artist_group=frame.artist.map(normalize_artist)).groupby('artist_group').agg(
        songs=('label','size'), year_end_hits=('label','sum')).to_csv(args.out/'group_distribution.csv')
    print(json.dumps({'out':str(args.out),'rows':len(frame),'source_sha256':ctx.source_sha256}))


if __name__ == '__main__': main()
