"""Verify positive entries; require complete release-to-cutoff history for negatives."""
from datetime import date
from . import LABEL_VERSION
from .common import cli, digest, parser, read, write
from .schema import candidate, iso, protocol, require
from .sources import validate_archive, weeks_between


def classify(r, archive, config, coverage=None):
    candidate(r)
    protocol(config)
    coverage = coverage or validate_archive(archive)
    cutoff = iso(config['chart_cutoff'])
    release = iso(r['first_release_date'])
    out = {**r, 'label': None, 'label_status': 'unknown', 'label_rule_version': LABEL_VERSION,
           'chart_name': 'Billboard Hot 100', 'chart_region': 'US', 'archive_cutoff_date': cutoff.isoformat(),
           'chart_coverage_snapshot_id': archive['snapshot_id'], 'chart_observation_start': release.isoformat(),
           'chart_observation_end': cutoff.isoformat(), 'chart_coverage_complete': False,
           'chart_entry_dates': [], 'chart_entry_evidence_urls': [], 'negative_absence_evidence': None,
           'label_reviewer': r['identity_reviewer'], 'label_reviewed_at': r['identity_reviewed_at'], 'excluded_reason': None}
    def unknown(reason):
        out['excluded_reason'] = reason
        return out
    if r['identity_status'] != 'verified' or r['version_policy'] != 'exact_recording_reviewed':
        return unknown('unresolved_identity_or_version')
    try:
        anniversary = release.replace(year=release.year + 2)
    except ValueError:
        anniversary = date(release.year + 2, 2, 28)
    if anniversary > cutoff:
        return unknown('insufficient_24_month_followup')
    required = weeks_between(release.isoformat(), cutoff.isoformat())
    out['chart_coverage_complete'] = bool(required) and set(required) <= set(coverage['covered_weeks'])
    hits, ambiguous, prerelease = [], False, False
    for week in archive['weeks']:
        if iso(week['date']) > cutoff:
            continue
        for e in week['entries']:
            if e['identity_status'] != 'verified':
                if iso(week['date']) >= release:
                    ambiguous = True  # absence cannot be proved while any weekly identity is unresolved
                continue
            exact = r['canonical_recording_id'] in e['canonical_recording_ids']
            same_work = r['canonical_work_id'] == e['canonical_work_id']
            if exact and (not same_work or not set(r['canonical_artist_ids']) & set(e['canonical_artist_ids'])):
                return unknown('canonical_identity_collision')
            if exact:
                if iso(week['date']) < release:
                    prerelease = True
                else:
                    hits.append(week)
            elif same_work and iso(week['date']) >= release:
                ambiguous = True
    if prerelease:
        return unknown('chart_predates_first_release_review_required')
    if hits:
        out.update(label=1, label_status='verified_charted', chart_entry_dates=sorted({w['date'] for w in hits}),
                   chart_entry_evidence_urls=sorted({w['evidence_url'] for w in hits}))
        return out
    if not out['chart_coverage_complete']:
        return unknown('incomplete_chart_coverage')
    if ambiguous:
        return unknown('unresolved_chart_identity_or_related_version')
    out.update(label=0, label_status='verified_noncharted', negative_absence_evidence={
        'archive_hash': digest(archive), 'query_recording_id': r['canonical_recording_id'],
        'query_work_id': r['canonical_work_id'], 'weeks': required, 'weeks_sha256': digest(required),
        'all_entries_identity_reviewed': True, 'query_rule': LABEL_VERSION})
    return out


def label_all(records, archive, config):
    require(len({r['recording_id'] for r in records}) == len(records), 'Duplicate candidate ID')
    require(len({r['canonical_recording_id'] for r in records}) == len(records), 'Duplicate canonical recording')
    coverage = validate_archive(archive)
    return [classify(r, archive, config, coverage) for r in sorted(records, key=lambda r: r['recording_id'])]


def main():
    p = parser(__doc__)
    p.add_argument('--archive', required=True); p.add_argument('--candidates', required=True)
    p.add_argument('--config', required=True); p.add_argument('--out')
    a = p.parse_args()
    rows = label_all(read(a.candidates), read(a.archive), read(a.config))
    if not a.dry_run:
        require(a.out, '--out required'); write(a.out, rows)
    from collections import Counter
    print(dict(Counter(x['label_status'] for x in rows)))


if __name__ == '__main__':
    cli(main)
