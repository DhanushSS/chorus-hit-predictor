"""Build a new immutable feature package from authorized local audio and chart exports."""
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from chorus_hit.config import FEATURE_COLUMNS, ROOT
from . import TASK, EXTRACTOR, LABEL_VERSION
from .audio import contract, extract
from .audit import summary
from .chart_membership import label_all
from .common import cli, code_identity, digest, environment, parser, read, runtime_check, seal, sha, stage, verify, write
from .matching import match_candidates
from .schema import protocol, require
from .sources import validate_archive


def safe_output(path, synthetic=False):
    p = Path(path).resolve()
    if p.is_relative_to(ROOT):
        require(p.is_relative_to(ROOT / 'output/hit_nonhit'), 'Dataset/split/test fixtures inside repo must use output/hit_nonhit/')
    return p


def build(archive, candidates, config, audio_root, destination, dry_run=False):
    runtime_check(); protocol(config)
    coverage = validate_archive(archive)
    require(config['chart_cutoff'] in coverage['covered_weeks'], 'Cutoff chart issue is not verified complete')
    rows = match_candidates(label_all(candidates, archive, config))
    synthetic = archive['synthetic']
    require(all(r.get('synthetic') is synthetic for r in candidates), 'Mixed real/synthetic candidate data')
    out = safe_output(destination, synthetic)
    if dry_run:
        return {'synthetic': synthetic, 'labels': dict(Counter(r['label_status'] for r in rows)), 'coverage': coverage, 'will_write': False}
    audio_root = Path(audio_root).resolve()
    features, eligible, errors = [], [], []
    with stage(out) as tmp:
        for row in rows:
            if row['label'] is None:
                continue
            try:
                require(row['match_level'] != 'unmatched', 'No comparable charted recording in artist/era')
                if config.get('evidence_policy') == 'public_song_query_v1':
                    require(row['version_policy'] == 'exact_recording_reviewed', 'Supplied audio must match the reviewed album recording')
                for key in ['audio_source_type', 'audio_rights_basis', 'audio_permission_evidence', 'audio_rights_reviewer']:
                    require(isinstance(row.get(key), str) and bool(row[key].strip()), f'Missing audio rights: {key}')
                require(row.get('audio_processing_authorized') is True, 'Audio processing not authorized')
                require(row.get('duplicate_review_status') == 'reviewed' and row.get('duplicate_review_evidence'), 'Audio duplicate/version review required')
                path = (audio_root / row['audio_path']).resolve()
                require(path.is_relative_to(audio_root), 'Audio path escapes authorized root')
                require(sha(path) == row.get('audio_sha256'), 'Audio hash differs from authorized recording')
                X, _, meta = extract(path, row.get('segment_start_seconds'), row.get('chorus_annotation'))
                eligible.append({**row, **meta, 'software_commit_sha': code_identity()['git_commit']})
                features.append(X.iloc[0].tolist())
            except (ValueError, OSError, KeyError, RuntimeError) as e:
                errors.append({'recording_id': row['recording_id'], 'label': row['label'], 'reason': str(e)})
        table = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        table.insert(0, 'recording_id', [r['recording_id'] for r in eligible])
        table.to_csv(tmp / 'features.csv', index=False, float_format='%.17g')
        write(tmp / 'records.json', eligible); write(tmp / 'candidate_labels.json', rows)
        write(tmp / 'candidates.json', candidates); write(tmp / 'archive.json', archive); write(tmp / 'config.json', config)
        write(tmp / 'audio_failures.json', errors)
        from .identity import review_queue
        write(tmp / 'identity_review_queue.json', review_queue(rows))
        audit = summary(rows, eligible)
        audit['audio_failures'] = errors
        audit['extraction_failure_rates'] = {str(y): {'failed': sum(e['label'] == y for e in errors), 'attempted': sum(r['label'] == y for r in rows)} for y in [0, 1]}
        for counts in audit['extraction_failure_rates'].values():
            counts['rate'] = counts['failed'] / counts['attempted'] if counts['attempted'] else None
        write(tmp / 'dataset_audit.json', audit)
        seal(tmp, {'kind': 'dataset', 'synthetic': synthetic, 'label_definition_version': LABEL_VERSION,
                   'chart_cutoff': config['chart_cutoff'], 'extractor': contract(), 'dataset_hash': digest(eligible),
                   'coverage': coverage, 'pinned_env': environment(), 'code': code_identity()})
    return audit


def load_dataset(folder):
    m = verify(folder, 'dataset'); folder = Path(folder)
    require(m['extractor'] == contract() and m['pinned_env'] == environment(), 'Extractor/environment mismatch')
    rows = read(folder / 'records.json'); config = read(folder / 'config.json')
    require(digest(rows) == m['dataset_hash'], 'Dataset record hash mismatch')
    archive = read(folder / 'archive.json'); candidates = read(folder / 'candidates.json')
    require(m['synthetic'] is archive['synthetic'] and all(r.get('synthetic') is m['synthetic'] for r in candidates), 'Synthetic/real provenance mismatch')
    require(m['chart_cutoff'] == config['chart_cutoff'] and m['label_definition_version'] == LABEL_VERSION, 'Dataset cutoff/label mismatch')
    recomputed = match_candidates(label_all(candidates, archive, config))
    by_id = {r['recording_id']: r for r in recomputed}
    for r in rows:
        require(r['recording_id'] in by_id and r['label'] in (0, 1), 'Unverified training label')
        for k, v in by_id[r['recording_id']].items():
            if k not in {'segment_start_seconds'}:
                require(r[k] == v, f'Label/identity evidence mismatch: {k}')
        require(r['extractor_version'] == EXTRACTOR and r['extractor_config_hash'] == digest(contract()), 'Mixed extractors')
        require(r['segment_samples'] == 330750 and r['sample_rate'] == 22050, 'Segment mismatch')
    X = pd.read_csv(folder / 'features.csv', float_precision='round_trip')
    require(list(X.columns) == ['recording_id'] + FEATURE_COLUMNS, 'Audio-only ordered feature schema required')
    require(X.recording_id.tolist() == [r['recording_id'] for r in rows], 'Feature membership/order mismatch')
    require(np.isfinite(X[FEATURE_COLUMNS].to_numpy()).all(), 'Nonfinite features')
    for r, v in zip(rows, X[FEATURE_COLUMNS].to_numpy()):
        require(r['feature_row_hash'] == digest(v.tolist()), 'Feature row hash mismatch')
    for key in ['audio_sha256', 'canonical_recording_id', 'canonical_work_id', 'feature_row_hash']:
        seen = {}
        for r in rows:
            require(r[key] not in seen or seen[r[key]] == r['label'], f'Conflicting labels for duplicate {key}')
            seen[r[key]] = r['label']
    return X, rows, m, config


def main():
    p = parser(__doc__)
    for flag in ['archive', 'candidates', 'config', 'audio-root', 'out']:
        p.add_argument('--' + flag, required=True)
    a = p.parse_args()
    print(build(read(a.archive), read(a.candidates), read(a.config), a.audio_root, a.out, a.dry_run))


if __name__ == '__main__':
    cli(main)
