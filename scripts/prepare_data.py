"""Verify the exact legacy source, or import an explicitly versioned alternate."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import os
from urllib.request import urlopen

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chorus_hit.config import DATA_PATH, FEATURE_COLUMNS, SOURCE_COMMIT, SOURCE_REPO
from chorus_hit.audit import LEGACY_DATASET, atomic_json
from chorus_hit.data import normalize_artist

URL = f'https://raw.githubusercontent.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus/{SOURCE_COMMIT}/Devolp/Final%20Data.csv'
SOURCE_SHA256 = 'b1e12d084dc503bf5e71c28c406bbc33e30bcc1f9000250b293d1b2b32851f0a'


def prepare(raw, alternate=None):
    digest = hashlib.sha256(raw).hexdigest()
    if alternate is None and digest != SOURCE_SHA256:
        raise ValueError('Source SHA-256 differs from pinned legacy CSV; alternate provenance is required')
    source = pd.read_csv(io.BytesIO(raw), float_precision='round_trip')
    if source.columns[5:].tolist() != FEATURE_COLUMNS:
        raise ValueError('Unexpected ordered source feature schema')
    if alternate is None and source.shape != (751, 523):
        raise ValueError('Pinned source shape changed')
    frame = source[['Artist', 'Title', 'Label', *FEATURE_COLUMNS]].rename(
        columns={'Artist':'artist','Title':'title','Label':'label'})
    if alternate is None:
        ids = [f'CH{i:04d}' for i in range(1,len(frame)+1)]
        provenance = {'dataset_version':LEGACY_DATASET,'source_repository':SOURCE_REPO,
            'source_commit':SOURCE_COMMIT,'source_file':'Devolp/Final Data.csv','source_url':URL,
            'license':'Apache-2.0; see licenses/DATASET_APACHE_2_0.txt',
            'label_definition':'1: source year-end Hot 100 hit; 0: other sampled weekly Hot 100 song',
            'evidence_status':'inherited; full independent audit unavailable',
            'id_policy':'Immutable legacy row IDs; exact source hash required'}
    else:
        required={'dataset_version','source_name','source_url','label_definition','recording_version','license','evidence_status'}
        if not required.issubset(alternate) or any(not alternate[k] for k in required):
            raise ValueError(f'Alternate provenance requires {sorted(required)}')
        if alternate['dataset_version']==LEGACY_DATASET or 'source_commit' in alternate:
            raise ValueError('Alternate source cannot claim the pinned legacy version or commit')
        provenance=dict(alternate)
        def stable_id(row):
            identity=[alternate['dataset_version'],normalize_artist(row.artist),normalize_artist(row.title),alternate['recording_version']]
            return 'REC-'+hashlib.sha256(json.dumps(identity,ensure_ascii=False).encode()).hexdigest()[:24]
        ids=[stable_id(row) for row in frame.itertuples()]
        if len(set(ids)) != len(ids):
            raise ValueError('Ambiguous duplicate identities; resolve recording versions before import')
        provenance['id_policy']='SHA-256 of version, normalized artist/title, declared recording version; independent of row order'
    frame.insert(0,'track_id',ids)
    provenance.update(source_sha256=digest,rows=len(frame),audio_features=len(FEATURE_COLUMNS),
        feature_provenance='Imported precomputed features; waveform parity unverified')
    return frame,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path); p.add_argument('--provenance',type=Path)
    p.add_argument('--out-dir',type=Path, help='Required new directory for alternate sources')
    a=p.parse_args()
    if a.provenance and (not a.source or not a.out_dir):
        p.error('Alternate import requires --source, --provenance and --out-dir')
    if a.source: raw=a.source.read_bytes()
    else:
        with urlopen(URL,timeout=60) as response: raw=response.read()
    frame,provenance=prepare(raw,json.loads(a.provenance.read_text()) if a.provenance else None)
    out=(a.out_dir or DATA_PATH.parent).resolve()
    csv=frame.to_csv(index=False).encode(); provenance['prepared_sha256']=hashlib.sha256(csv).hexdigest()
    if out.exists():
        if not a.provenance and (out/'chorus_features.csv').is_file() and (out/'chorus_features.csv').read_bytes()==csv:
            print('Exact legacy data already present; preserved existing files'); return
        p.error('Output directory already exists; imports never overwrite an existing dataset')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.import-',dir=out.parent) as tmp:
        staging=Path(tmp)/'dataset'; staging.mkdir()
        (staging/'chorus_features.csv').write_bytes(csv); atomic_json(staging/'provenance.json',provenance)
        from chorus_hit.data import load_data
        load_data(staging/'chorus_features.csv')
        os.rename(staging,out)
    print(f'Prepared {len(frame)} songs: {out}')


if __name__=='__main__': main()
