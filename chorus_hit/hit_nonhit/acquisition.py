"""Fixed-cohort acquisition ledger and feature-availability audit; never downloads audio."""
import argparse
from collections import Counter
from pathlib import Path
import time

import pandas as pd
import requests

from chorus_hit.config import ROOT
from .common import cli, read, sha, stage, stamp, write
from .matching import match_candidates
from .schema import require


COHORT = ROOT / 'data/hit_nonhit_v1/audio_intake.csv'
COHORT_SHA256 = 'c5414777f3aca769e493f8eaac130052bd05d7c3b2eca429ae48791f6d4e4a98'
LEDGER_FIELDS = [
    'recording_id', 'title', 'artist', 'album', 'proposed_label', 'match_set_id',
    'candidate_source_url', 'source_type', 'permission_basis', 'permission_reference',
    'license_review_status', 'exact_recording_version_status', 'download_permitted',
    'analysis_permitted', 'local_audio_path', 'format', 'duration_sec', 'sha256',
    'chorus_review_status', 'disposition', 'rejection_reason',
]


def selected_cohort():
    require(sha(COHORT) == COHORT_SHA256, 'Fixed 278-song cohort changed; scope review required')
    return pd.read_csv(COHORT, dtype=str, keep_default_na=False)


def validate_selected(table):
    """Allow a subset, but never expand or relabel the approved 278 records."""
    require({'recording_id', 'proposed_label'} <= set(table.columns), 'Missing cohort identity/label columns')
    require(table.recording_id.is_unique, 'Duplicate recording IDs in audio intake')
    labels = selected_cohort().set_index('recording_id').proposed_label.to_dict()
    for entry in table.to_dict('records'):
        require(entry['recording_id'] in labels, 'Recording outside fixed 278-song cohort')
        require(str(entry['proposed_label']) == labels[entry['recording_id']], 'Edited fixed-cohort label')


def initialize(destination):
    """Create an empty LOCAL ledger without inventing permissions or recording matches."""
    table = selected_cohort()
    candidates = read(ROOT / 'data/hit_nonhit_v1/academic/candidates.json')
    labels = table.set_index('recording_id').proposed_label.to_dict()
    matched = match_candidates([{**r, 'label': int(labels[r['recording_id']])}
                                for r in candidates if r['recording_id'] in labels])
    matches = {r['recording_id']: r['match_set_id'] for r in matched}
    rows = []
    for record in table.to_dict('records'):
        row = dict.fromkeys(LEDGER_FIELDS, '')
        row.update({k: record[k] for k in ['recording_id', 'title', 'artist', 'album', 'proposed_label']})
        row.update(match_set_id=matches[record['recording_id']], source_type='no_verified_audio_source',
                   license_review_status='unverified', exact_recording_version_status='unverified',
                   download_permitted='unknown', analysis_permitted='unknown', chorus_review_status='unreviewed',
                   disposition='MISSING', rejection_reason='No permitted exact-recording file identified; metadata is not audio')
        rows.append(row)
    # This is a blank acquisition template, never a claim about files elsewhere.
    summary = {'created_at': stamp(), 'cohort_sha256': COHORT_SHA256, 'selected_metadata': len(table),
               'metadata_class_counts': dict(Counter(table.proposed_label)), 'ledger_files_supplied': 0,
               'ledger_status_counts': {'MISSING': len(table)}, 'scope': 'blank local acquisition template'}
    with stage(destination) as temp:
        pd.DataFrame(rows, columns=LEDGER_FIELDS).to_csv(temp / 'acquisition_ledger_278.csv', index=False)
        table.to_csv(temp / 'audio_intake_filled_278.csv', index=False)
        write(temp / 'summary.json', summary)
    return summary


def parse_counts(payload, ids):
    """An absent ID in a successful count response means no submission; errors stay unknown."""
    require(isinstance(payload, dict) and set(payload) <= set(ids) | {'mbid_mapping'}, 'Unexpected feature count response')
    require('error' not in payload, 'Feature count service error')
    mapping = payload.get('mbid_mapping', {})
    require(isinstance(mapping, dict), 'Invalid recording mapping')
    result = {}
    for identity in ids:
        if identity in mapping:
            result[identity] = None  # merged IDs need explicit identity review
            continue
        entry = payload.get(identity, {'count': 0})
        require(isinstance(entry, dict) and set(entry) == {'count'}, 'Invalid feature submission entry')
        value = entry['count']
        require(type(value) is int and value >= 0, 'Invalid feature submission count')
        result[identity] = value
    return result


def probe_features(destination, session=None, sleep=time.sleep):
    """Audit exact MBIDs through the documented CC0 AcousticBrainz bulk-count API.

    Counts are not files, compatible chorus vectors, or reviewed recording identities.
    This does not build a training table or change the 518-feature experiment.
    """
    table = selected_cohort()
    client = session or requests.Session()
    client.headers.update({'User-Agent': 'ChorusHitPredictor-AcademicFeasibility/1.0'})
    rows, requests_log = [], []
    with stage(destination) as temp:
        for start in range(0, len(table), 25):
            batch = table.iloc[start:start + 25]
            ids = [r.removeprefix('mb-') for r in batch.recording_id]
            record = {'recording_ids': ids, 'checked_at': stamp()}
            counts = dict.fromkeys(ids)
            wait = 1.1
            try:
                response = client.get('https://acousticbrainz.org/api/v1/count',
                                      params={'recording_ids': ';'.join(ids)}, timeout=25)
                record['http_status'] = response.status_code
                if response.status_code == 429:
                    wait = 61  # stop this probe; a later explicit run can retry
                response.raise_for_status()
                payload = response.json()
                counts = parse_counts(payload, ids)
                record['response'] = payload
                if response.headers.get('X-RateLimit-Remaining') == '0':
                    wait = max(wait, float(response.headers.get('X-RateLimit-Reset-In', '10')))
            except (requests.RequestException, ValueError, TypeError) as error:
                # Do not serialize exception text: URLs can contain credentials in custom clients.
                record['error_type'] = type(error).__name__
            requests_log.append(record)
            for row, identity in zip(batch.to_dict('records'), ids):
                count = counts[identity]
                rows.append({'recording_id': row['recording_id'], 'proposed_label': row['proposed_label'],
                             'submissions': count,
                             'status': 'UNKNOWN' if count is None else 'FEATURES_PRESENT' if count > 0 else 'NO_SUBMISSION'})
            if wait > 60 or not (wait >= 0):
                # Respect a long/invalid cooldown without sleeping indefinitely or retrying early.
                remaining = table.iloc[start + len(batch):]
                rows.extend({'recording_id': r.recording_id, 'proposed_label': r.proposed_label,
                             'submissions': None, 'status': 'UNKNOWN'} for r in remaining.itertuples())
                break
            sleep(wait)
        present = [r for r in rows if r['status'] == 'FEATURES_PRESENT']
        summary = {'created_at': stamp(), 'cohort_sha256': COHORT_SHA256, 'queried_rows': sum(len(r['recording_ids']) for r in requests_log),
                   'status_counts': dict(Counter(r['status'] for r in rows)),
                   'features_present_class_counts': dict(Counter(r['proposed_label'] for r in present)),
                   'source': 'https://acousticbrainz.org/', 'source_license': 'CC0',
                   'compatible_518_chorus_rows': 0, 'audio_files_downloaded': 0,
                   'scope': 'Availability only. Essentia whole-recording features; version/identity review still required.'}
        write(temp / 'requests.json', requests_log)
        write(temp / 'coverage.json', rows)
        write(temp / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['init', 'probe-features'])
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(initialize(a.out) if a.action == 'init' else probe_features(a.out))


if __name__ == '__main__':
    cli(main)
