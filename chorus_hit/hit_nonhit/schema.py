"""Strict evidence inputs; unresolved identity is allowed only in the audit queue."""
from datetime import date
import re
from . import TASK, LABEL_VERSION

TEXT_FIELDS = ('recording_id', 'canonical_work_id', 'canonical_recording_id', 'album_id', 'track_id',
               'title', 'original_artist_credit', 'recording_version', 'first_release_date',
               'album_release_date', 'market', 'identity_source', 'identity_status', 'identity_reviewer',
               'identity_reviewed_at', 'version_policy', 'source_license_snapshot', 'created_at', 'source_checked_at')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def iso(value):
    require(isinstance(value, str) and bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', value)), 'Dates need exact YYYY-MM-DD precision')
    return date.fromisoformat(value)


def strings(values):
    return isinstance(values, list) and bool(values) and all(isinstance(x, str) and x.strip() for x in values) and len(values) == len(set(values))


def candidate(r):
    require(r.get('dataset_version') == TASK, 'Wrong dataset version; legacy rows cannot be relabeled')
    for key in TEXT_FIELDS:
        require(isinstance(r.get(key), str) and bool(r[key].strip()), f'Missing candidate evidence: {key}')
    require(strings(r.get('canonical_artist_ids')), 'Canonical artist IDs required')
    require(strings(r.get('identity_evidence_urls')), 'Identity evidence URLs required')
    require(type(r.get('track_number')) is int and r['track_number'] > 0, 'Track number must be positive')
    require(r['identity_status'] in {'verified', 'ambiguous', 'pending'}, 'Invalid identity status')
    require(r['market'] == 'US', 'Only US recordings/cohort supported')
    release = iso(r['first_release_date'])
    require(release <= iso(r['album_release_date']), 'First release must include earlier single/promotional release')
    return r


def source(s, payload_hash, synthetic=False):
    for k in ['provider', 'license', 'license_snapshot', 'permission_evidence', 'reviewer', 'checked_at', 'export_sha256']:
        require(isinstance(s.get(k), str) and bool(s[k].strip()), f'Missing source provenance: {k}')
    require(s.get('research_use_authorized') is True, 'Source permission not confirmed')
    require(s['export_sha256'] == payload_hash, 'Source export hash mismatch')
    require(s.get('synthetic') is synthetic, 'Synthetic source declaration mismatch')


def protocol(c, allow_pending=False):
    require(c.get('task_id') == TASK and c.get('label_definition_version') == LABEL_VERSION, 'Task/label definition mismatch')
    require(c.get('chart_name') == 'Billboard Hot 100' and c.get('chart_region') == 'US', 'Wrong chart/region')
    require(c.get('minimum_followup_months') == 24, 'This protocol requires 24 months minimum follow-up')
    cutoff = c.get('chart_cutoff')
    if cutoff is None and allow_pending:
        return c
    require(cutoff is not None, 'No verified chart cutoff; real_data_training_status=blocked_data')
    require(iso(cutoff).weekday() == 5, 'Cutoff must be a Saturday chart issue date for this protocol')
    return c
