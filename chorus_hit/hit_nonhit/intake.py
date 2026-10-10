"""Convert the simple acquisition CSV into reviewed local-audio extraction inputs."""
import argparse
import math
from collections import Counter
from pathlib import Path
import pandas as pd
from chorus_hit.config import ROOT
from .academic_evidence import public_label_all
from .build_dataset import build
from .common import cli, read, sha, stamp
from .matching import match_candidates
from .schema import require


def yes(value):
    return str(value).strip().lower() in {'yes', 'true', '1'}


def review(intake, candidates, archive, config, audio_root):
    table = pd.read_csv(intake, dtype=str, keep_default_na=False)
    require({'recording_id', 'proposed_label', 'audio_path', 'permission_basis', 'permission_reference',
             'reviewer', 'recording_matches', 'chorus_start_seconds', 'chorus_confirmed'} <= set(table.columns), 'Incomplete audio intake CSV')
    require(table.recording_id.is_unique, 'Duplicate recording IDs in audio intake')
    labels = match_candidates(public_label_all(candidates, archive, config))
    by_id = {r['recording_id']: r for r in labels}
    originals = {r['recording_id']: r for r in candidates}
    root = Path(audio_root).resolve(); ready, errors = [], []
    for entry in table.to_dict('records'):
        identity = entry['recording_id']
        require(identity in by_id, 'Unknown recording ID in audio intake')
        row = by_id[identity]
        require(row['label'] is not None and entry['proposed_label'] == str(row['label']), 'Edited/unverified proposed label')
        if not entry['audio_path']:
            errors.append({'recording_id': identity, 'reason': 'audio_path not supplied'}); continue
        try:
            require(row['match_level'] != 'unmatched', 'No comparable positive for this recording')
            for field in ['permission_basis', 'permission_reference', 'reviewer']:
                require(bool(entry[field].strip()), f'Missing {field}')
            require(yes(entry['recording_matches']), 'Listen/check that the file is the named studio album recording, then set recording_matches=yes')
            path = (root / entry['audio_path']).resolve()
            require(path.is_relative_to(root), 'Audio path escapes supplied root')
            require(path.is_file(), 'Audio file not found')
            require(path.suffix.lower() in {'.wav', '.flac', '.mp3', '.ogg'}, 'Unsupported local audio type')
            start = float(entry['chorus_start_seconds']) if entry['chorus_start_seconds'] else None
            require(start is None or (math.isfinite(start) and start >= 0), 'Chorus start must be finite and nonnegative')
            annotation = None
            if yes(entry['chorus_confirmed']):
                require(start is not None, 'Confirmed chorus needs its start time in seconds')
                annotation = {'reviewer': entry['reviewer'], 'evidence': 'Listened to supplied recording; chorus start recorded in audio intake', 'is_chorus': True, 'start_seconds': start}
            ready.append({**originals[identity], 'version_policy': 'exact_recording_reviewed',
                          'audio_path': str(path.relative_to(root)), 'audio_sha256': sha(path),
                          'audio_processing_authorized': True, 'audio_source_type': 'local_permitted_recording',
                          'audio_rights_basis': entry['permission_basis'], 'audio_permission_evidence': entry['permission_reference'],
                          'audio_rights_reviewer': entry['reviewer'], 'audio_reviewed_at': stamp(),
                          'duplicate_review_status': 'reviewed', 'duplicate_review_evidence': 'Intake reviewer confirmed the named studio album recording',
                          'segment_start_seconds': start, 'chorus_annotation': annotation})
        except (ValueError, OSError) as error:
            errors.append({'recording_id': identity, 'reason': str(error)})
    # Do not keep a control whose positive never received usable input.
    ready_labels = match_candidates(public_label_all(ready, archive, config))
    good_ids = {r['recording_id'] for r in ready_labels if r['match_level'] != 'unmatched'}
    dropped = [r for r in ready if r['recording_id'] not in good_ids]
    errors.extend({'recording_id': r['recording_id'], 'reason': 'Matching positive audio is not supplied'} for r in dropped)
    ready = [r for r in ready if r['recording_id'] in good_ids]
    return ready, {'intake_rows': len(table), 'ready_audio_rows': len(ready),
                   'ready_class_counts': dict(Counter(str(by_id[r['recording_id']]['label']) for r in ready)),
                   'missing_or_invalid': errors, 'features_reused_from_legacy': False}


def ingest(prepared, discovered, intake, audio_root, destination=None, dry_run=False):
    # The discovery pool is larger than the approved acquisition cohort.
    # Enforce scope before reading audio or constructing extraction artifacts.
    from .acquisition import validate_selected
    validate_selected(pd.read_csv(intake, dtype=str, keep_default_na=False))
    archive = read(Path(prepared) / 'archive.json'); candidates = read(Path(discovered) / 'candidates.json')
    config = read(ROOT / 'configs/hit_nonhit_academic.json')
    ready, report = review(intake, candidates, archive, config, audio_root)
    if dry_run: return report
    require(set(report['ready_class_counts']) == {'0', '1'}, 'Permitted matched audio for BOTH classes is required; use --dry-run for the missing-file list')
    require(destination, '--out required for extraction')
    return {'intake': report, 'dataset': build(archive, ready, config, audio_root, destination)}


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ['prepared', 'discovered', 'intake', 'audio-root']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--out'); p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(); print(ingest(a.prepared, a.discovered, a.intake, a.audio_root, a.out, a.dry_run))


if __name__ == '__main__': cli(main)
