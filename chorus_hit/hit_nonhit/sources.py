"""Authorized local weekly exports only. No scraping, credentials or audio downloading."""
from datetime import timedelta
from .common import digest, read
from .schema import iso, require, source, strings


def weeks_between(start, end):
    start, end = iso(start), iso(end)
    current = start + timedelta(days=(5 - start.weekday()) % 7)
    result = []
    while current <= end:
        result.append(current.isoformat())
        current += timedelta(days=7)
    return result


def validate_archive(a):
    require(a.get('chart_name') == 'Billboard Hot 100' and a.get('chart_region') == 'US', 'Wrong chart/region')
    require(type(a.get('synthetic')) is bool, 'Declare synthetic true/false')
    source(a['source'], digest(a['weeks']), a['synthetic'])
    require(bool(a.get('snapshot_id')), 'Archive snapshot ID required')
    require(iso(a['start']) <= iso(a['end']), 'Reversed archive dates')
    expected = set(weeks_between(a['start'], a['end']))
    dates = [w['date'] for w in a['weeks']]
    require(len(dates) == len(set(dates)), 'Duplicate chart weeks')
    require(set(dates) <= expected, 'Unexpected chart date or weekday')
    complete, incomplete = [], []
    for w in a['weeks']:
        require(bool(w.get('evidence_url')) and bool(w.get('reviewer')), 'Week evidence/reviewer missing')
        require(w.get('entries_sha256') == digest(w['entries']), 'Chart entries hash mismatch')
        ranks = [e['rank'] for e in w['entries']]
        require(all(type(x) is int and 1 <= x <= 100 for x in ranks) and len(ranks) == len(set(ranks)), 'Invalid/duplicate ranks')
        keys = []
        for e in w['entries']:
            require(e.get('title') and e.get('artist_credit'), 'Raw chart identity missing')
            require(e.get('identity_status') in {'verified', 'unresolved'}, 'Chart identity status missing')
            if e['identity_status'] == 'verified':
                require(e.get('canonical_work_id') and strings(e.get('canonical_recording_ids')) and strings(e.get('canonical_artist_ids')), 'Chart canonical identity missing')
                require(e.get('identity_evidence') and e.get('identity_reviewer') and e.get('version_reviewed') is True, 'Chart version/identity review missing')
                keys.extend(e['canonical_recording_ids'])
        require(len(keys) == len(set(keys)), 'Duplicate recording in one chart issue')
        (complete if set(ranks) == set(range(1, 101)) else incomplete).append(w['date'])
    return {'snapshot_id': a['snapshot_id'], 'archive_hash': digest(a), 'covered_weeks': sorted(complete),
            'missing_or_incomplete_weeks': sorted(expected - set(complete)), 'incomplete_weeks': incomplete,
            'source_permission_assertion_checked': True, 'source_authority_independently_authenticated': False}


def load_archive(path):
    a = read(path)
    return a, validate_archive(a)
