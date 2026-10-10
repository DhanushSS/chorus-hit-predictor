"""Audit HSP-S, allow only scalar audio descriptors, and freeze connected groups once."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit
from chorus_hit.hit_nonhit.audit import connected_groups
from chorus_hit.hit_nonhit.common import digest, environment, read, sha, stage, stamp, write
from chorus_hit.hit_nonhit.freeze_split import grouped_folds
from chorus_hit.hit_nonhit.identity import normalize
from chorus_hit.hit_nonhit.schema import require
from . import TASK, SOURCE_MD5, SOURCE_URL
from .download import verify_source


def audio_columns(schema):
    return sorted(f.name for f in schema if f.name.startswith(('lowlevel.', 'rhythm.', 'tonal.'))
                  and (pa.types.is_floating(f.type) or pa.types.is_integer(f.type)))


def strings(value):
    if value is None: return []
    if isinstance(value, str): return [value] if value.strip() else []
    if isinstance(value, (np.ndarray, list, tuple)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return []


def label(peak, weeks):
    if pd.isna(peak) and pd.isna(weeks): return 0
    if pd.notna(peak) and pd.notna(weeks) and float(peak).is_integer() and 1 <= peak <= 100 and float(weeks).is_integer() and weeks >= 1: return 1
    return None


def identity(row, vector):
    artists = set()
    generic = {'various artists', 'unknown', 'unknown artist', 'anonymous', 'various'}
    for suffix in ['ll', 'hl']:
        artists.update('mb:' + s.casefold() for s in strings(row.get(f'metadata.tags.musicbrainz_artistid_{suffix}'))
                       if s.casefold() != '89ad4ac3-39f7-470e-963a-56509c546377')
        artists.update('name:' + normalize(s) for s in strings(row.get(f'metadata.tags.artist_{suffix}'))
                       if normalize(s) and normalize(s) not in generic)
    require(bool(artists), 'Missing usable recording-artist identity')
    title = strings(row.get('metadata.tags.title_ll'))
    # A conservative title/credit proxy connects other recordings of the same named work.
    work = digest([normalize(title[0]), sorted(artists)]) if title else None
    return {'recording_id': str(row['uuid']), 'canonical_recording_id': str(row['mbid']),
            'canonical_artist_ids': sorted(artists), 'canonical_work_id': work,
            'album_id': next(iter(strings(row.get('metadata.tags.musicbrainz_albumid_ll'))), None),
            'dedupe_group_id': next(('encoded-md5:' + s for s in strings(row.get('metadata.audio_properties.md5_encoded_ll'))), None),
            'feature_row_hash': digest([float(v) if np.isfinite(v) else None for v in vector])}


def seal(folder, kind, extra):
    folder = Path(folder)
    files = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    value = {'task_id': TASK, 'kind': kind, 'files': files, **extra}
    write(folder / 'manifest.json', value)
    return value


def verify(folder, kind):
    folder = Path(folder); manifest = read(folder / 'manifest.json')
    require(manifest.get('task_id') == TASK and manifest.get('kind') == kind, 'Wrong HSP-S artifact')
    required = {'split': {'development.parquet', 'locked.parquet', 'config.json', 'split.json', 'audit.json', 'columns.json', 'identity.json'},
                'model': {'model.joblib', 'summary.json', 'config.json', 'diagnostics.json'}}[kind]
    require(required <= set(manifest['files']), 'Manifest missing required HSP-S files')
    extras = {'manifest.json'} | ({'test_access.json'} if kind == 'split' else {'locked_evaluation.json'})
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    require(actual - extras == set(manifest['files']), 'Unexpected or missing HSP-S artifact files')
    for name, expected in manifest['files'].items():
        p = (folder / name).resolve()
        require(p.is_relative_to(folder.resolve()) and p.is_file() and sha(p) == expected, 'HSP-S artifact hash mismatch')
    return manifest


def prepare(source, destination, config):
    verify_source(source); cfg = read(config)
    require(cfg['task_id'] == TASK, 'Wrong experiment config')
    table = pq.ParquetFile(source); columns = audio_columns(table.schema_arrow)
    require(len(columns) == 436 and table.metadata.num_rows == 7736, 'Unexpected pinned HSP-S release schema')
    meta = ['uuid', 'mbid', 'peakPos', 'weeks'] + [n for n in table.schema_arrow.names if n.startswith('metadata.') and
            any(k in n for k in ['tags.artist_', 'tags.title_ll', 'musicbrainz_artistid_', 'musicbrainz_albumid_ll',
                                'audio_properties.md5_encoded_ll', 'audio_properties.codec_ll', 'audio_properties.sample_rate_ll',
                                'audio_properties.length_ll', 'metadata.version.essentia', 'audio_properties.analysis_sample_rate_ll'])]
    raw = table.read(columns=columns + meta).to_pandas().reset_index(drop=True)
    require(raw.uuid.notna().all() and raw.uuid.is_unique and raw.mbid.notna().all(), 'Missing/duplicate dataset identity')
    X = raw[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    records, accepted, excluded = [], [], []
    for i, row in raw.iterrows():
        y = label(row.peakPos, row.weeks)
        if y is None:
            excluded.append({'uuid': row.uuid, 'reason': 'Inconsistent chart outcome fields'}); continue
        if X.iloc[i].notna().sum() == 0:
            excluded.append({'uuid': row.uuid, 'reason': 'No finite audio descriptors'}); continue
        try: record = identity(row, X.iloc[i].to_numpy())
        except ValueError as error:
            excluded.append({'uuid': row.uuid, 'reason': str(error)}); continue
        records.append({**record, 'label': y}); accepted.append(i)
    # Conflicting identical vectors are ambiguous; otherwise retain one row by UUID.
    by_vector = {}
    for pos, record in enumerate(records): by_vector.setdefault(record['feature_row_hash'], []).append(pos)
    remove = set()
    for positions in by_vector.values():
        if len(positions) <= 1: continue
        conflict = len({records[p]['label'] for p in positions}) > 1
        keep = min(positions, key=lambda p: records[p]['recording_id'])
        for pos in positions:
            if conflict or pos != keep:
                remove.add(pos); excluded.append({'uuid': records[pos]['recording_id'], 'reason': 'Conflicting identical audio features' if conflict else 'Duplicate identical audio features'})
    accepted = [i for p, i in enumerate(accepted) if p not in remove]
    records = [r for p, r in enumerate(records) if p not in remove]
    # Conflicting labels for the same recording are not treated as independent examples.
    labels_by_mbid = {}
    for r in records: labels_by_mbid.setdefault(r['canonical_recording_id'], set()).add(r['label'])
    bad = {k for k, v in labels_by_mbid.items() if len(v) > 1}
    if bad:
        excluded.extend({'uuid': r['recording_id'], 'reason': 'Conflicting recording labels'} for r in records if r['canonical_recording_id'] in bad)
        accepted = [i for i, r in zip(accepted, records) if r['canonical_recording_id'] not in bad]
        records = [r for r in records if r['canonical_recording_id'] not in bad]
    groups = np.array(connected_groups(records)); y = np.array([r['label'] for r in records])
    require(len(set(groups)) >= 20, 'Insufficient independent artist-connected groups')
    frame = X.iloc[accepted].reset_index(drop=True)
    frame.insert(0, 'uuid', [r['recording_id'] for r in records]); frame.insert(1, 'label', y); frame.insert(2, 'group', groups)
    for target, source_name in [('codec', 'metadata.audio_properties.codec_ll'), ('sample_rate', 'metadata.audio_properties.sample_rate_ll'),
                                ('duration', 'metadata.audio_properties.length_ll'), ('extractor', 'metadata.version.essentia')]:
        values = raw.iloc[accepted][source_name].reset_index(drop=True)
        frame['nuisance_' + target] = values.fillna('unknown') if target in {'codec', 'extractor'} else pd.to_numeric(values, errors='coerce')
    e = cfg['evaluation']; dev, test = next(GroupShuffleSplit(n_splits=1, test_size=e['test_fraction'], random_state=e['seed']).split(frame, y, groups))
    for ix in [dev, test]:
        require(set(y[ix]) == {0, 1}, 'Fixed partition lacks a class; do not search seeds')
        require(all(len(set(groups[ix][y[ix] == k])) >= 5 for k in [0, 1]), 'Insufficient class/group support')
    outer = grouped_folds(y[dev], groups[dev], e['outer_folds'], e['seed'])
    for tr, _ in outer: grouped_folds(y[dev][tr], groups[dev][tr], e['inner_folds'], e['seed'])
    detail = {name: {'rows': len(ix), 'class_counts': dict(Counter(str(v) for v in y[ix])), 'groups': len(set(groups[ix])),
                     'ids': frame.uuid.iloc[ix].tolist()} for name, ix in [('development', dev), ('locked', test)]}
    audit = {'source_rows': len(raw), 'accepted': len(records), 'class_counts': dict(Counter(str(v) for v in y)), 'groups': len(set(groups)),
             'largest_group': max(Counter(groups).values()), 'exclusions': excluded, 'scalar_audio_features': len(columns),
             'missing_audio_cells': int(frame[columns].isna().sum().sum()),
             'label_scope': 'Publisher benchmark: chart fields present vs absent; not independently verified never-charted status',
             'unused_fields': 'Chart outcomes, listening counts, all metadata, high-level classifiers and variable-length audio vectors are excluded from X',
             'raw_audio_acquired': 0, 'chorus_features_extracted': 0, 'year_mapping_used': False}
    with stage(destination) as tmp:
        frame.iloc[dev].to_parquet(tmp / 'development.parquet', index=False)
        frame.iloc[test].to_parquet(tmp / 'locked.parquet', index=False)
        for name, value in [('config', cfg), ('columns', columns), ('split', detail), ('identity', records), ('audit', audit)]: write(tmp / f'{name}.json', value)
        seal(tmp, 'split', {'source_url': SOURCE_URL, 'source_md5': SOURCE_MD5, 'source_sha256': sha(source), 'created_at': stamp(),
                            'config_hash': digest(cfg), 'environment': environment(), 'feature_schema_hash': digest(columns), 'protocol': 'one fixed 20% group holdout; 5x3 nested group CV; no seed retries'})
    return {k: v for k, v in audit.items() if k != 'exclusions'}


def development(folder):
    m = verify(folder, 'split'); folder = Path(folder)
    cfg, columns, split = read(folder / 'config.json'), read(folder / 'columns.json'), read(folder / 'split.json')
    require(digest(cfg) == m['config_hash'] and digest(columns) == m['feature_schema_hash'], 'HSP-S schema/config mismatch')
    require(m['environment'] == environment(), 'Environment differs from frozen split')
    frame = pd.read_parquet(folder / 'development.parquet')
    require(frame.uuid.tolist() == split['development']['ids'], 'Development membership changed')
    require(not set(split['development']['ids']) & set(split['locked']['ids']), 'Split identity overlap')
    identities = read(folder / 'identity.json'); groups = dict(zip([r['recording_id'] for r in identities], connected_groups(identities)))
    require(not {groups[i] for i in split['development']['ids']} & {groups[i] for i in split['locked']['ids']}, 'Connected group overlap')
    require(frame.group.tolist() == [groups[i] for i in frame.uuid], 'Group membership changed')
    return frame, columns, cfg, m
