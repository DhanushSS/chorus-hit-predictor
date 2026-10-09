"""Academic song-level chart evidence, with complete windows and conservative ambiguity handling."""
from collections import defaultdict
from difflib import SequenceMatcher
import re
import unicodedata
from .common import digest
from .schema import require, iso
from .sources import weeks_between

COHORT = 'year_end_hits_vs_verified_noncharted_2006_2021'


def normalized(text):
    text = unicodedata.normalize('NFKD', str(text)).casefold().replace('&', ' and ')
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in text).split())


def credit_matches(credit, aliases):
    # Full credit or explicit lead-artist boundary; substring matches are unsafe.
    full = normalized(credit)
    lead = re.split(r'\s+(?:featuring|feat\.?|ft\.?)\s+', str(credit), flags=re.I)[0]
    return any(normalized(a) in {full, normalized(lead)} for a in aliases)


def title_base(title):
    return normalized(re.sub(r'\([^)]*\)|\[[^]]*\]', '', str(title)))


def alternate_version_title(title):
    qualifiers = re.findall(r'\(([^)]*)\)|\[([^]]*)\]|\s[-–—]\s(.+)$', str(title))
    return any(re.search(r'\b(live|remix|mix|demo|karaoke|instrumental|acoustic|interview|rehearsal|edit|clean)\b', ' '.join(parts), re.I)
               for parts in qualifiers)


class ChartIndex:
    def __init__(self, weekly, annual, coverage):
        self.by_title = defaultdict(list); self.annual_by_title = defaultdict(list)
        self.by_credit = defaultdict(list)
        for r in weekly:
            self.by_title[normalized(r['title'])].append(r)
            self.by_credit[normalized(r['artist'])].append(r)
        for r in annual:
            self.annual_by_title[normalized(r['title'])].append(r)
        self.weekly = weekly; self.coverage = coverage
        self._artist_rows = {}

    def hits(self, titles, artists, cutoff):
        return [r for title in titles for r in self.by_title[normalized(title)] if r['date'] <= cutoff and credit_matches(r['artist'], artists)]

    def yearend(self, titles, artists):
        return [r for title in titles for r in self.annual_by_title[normalized(title)] if credit_matches(r['artist'], artists)]

    def absence(self, titles, artists, release, cutoff):
        required = weeks_between(release, cutoff)
        missing = sorted(set(required) - set(self.coverage['covered_weeks']))
        if not required or missing:
            return {'status': 'unknown', 'reason': 'incomplete_weekly_window', 'missing_weeks': missing}
        hits = self.hits(titles, artists, cutoff)
        if hits:
            return {'status': 'charted', 'hits': hits}
        artist_key = tuple(sorted(artists))
        if artist_key not in self._artist_rows:
            # Include joint credits as ambiguity leads even when exact lead matching fails.
            keys = [normalized(a) for a in artists]
            self._artist_rows[artist_key] = [r for credit, rows in self.by_credit.items()
                                           if any((' ' + a + ' ') in (' ' + credit + ' ') for a in keys) for r in rows]
        ambiguous = []
        for r in self._artist_rows[artist_key]:
            if r['date'] > cutoff: continue
            for title in titles:
                base, other = title_base(title), title_base(r['title'])
                if base == other or (len(base) >= 5 and SequenceMatcher(None, normalized(title), normalized(r['title'])).ratio() >= .8):
                    ambiguous.append(r); break
        if ambiguous:
            return {'status': 'unknown', 'reason': 'similar_title_or_joint_credit_review', 'possible_entries': ambiguous}
        return {'status': 'verified_noncharted_in_public_archive', 'weeks_queried': len(required),
                'window_start': release, 'window_end': cutoff, 'covered_weeks_hash': digest(required),
                'title_queries': titles, 'artist_queries': artists,
                'limitation': 'Song-level absence in the audited public archive; not independently certified by Billboard and not a promise of never charting.'}


def coverage_audit(weekly, start, cutoff):
    expected = weeks_between(start, cutoff)
    groups = defaultdict(list)
    for row in weekly:
        if start <= row['date'] <= cutoff:
            require(iso(row['date']).weekday() == 5, 'Unexpected issue weekday in study window')
            require(all(isinstance(row.get(k), str) and row[k].strip() for k in ['title', 'artist']), 'Blank weekly identity')
            require(type(row.get('rank')) is int, 'Weekly rank must be an integer')
            groups[row['date']].append(row)
    complete, bad = [], {}
    for day, rows in groups.items():
        ranks = [r['rank'] for r in rows]
        pairs = [(normalized(r['title']), normalized(r['artist'])) for r in rows]
        if len(rows) != 100 or set(ranks) != set(range(1, 101)) or len(set(pairs)) != 100:
            bad[day] = {'rows': len(rows), 'distinct_ranks': len(set(ranks)), 'distinct_song_credits': len(set(pairs))}
        else: complete.append(day)
    return {'start': start, 'cutoff': cutoff, 'expected_weeks': len(expected), 'covered_weeks': sorted(complete),
            'missing_or_incomplete_weeks': sorted(set(expected) - set(complete)), 'bad_issues': bad,
            'authority': 'public compiled archive; rank/continuity checks do not authenticate every title'}


def validate_public_archive(archive):
    from .public_sources import PUBLIC_FILES
    require(archive.get('resolution_policy') == 'public_song_query_v1' and archive.get('cohort') == COHORT, 'Undeclared public archive policy')
    require(archive.get('synthetic') is False and archive.get('chart_name') == 'Billboard Hot 100' and archive.get('chart_region') == 'US', 'Invalid public archive task')
    require(archive.get('start') == '2000-01-01' and archive.get('end') == '2023-12-30', 'Public cohort window differs')
    require(archive['snapshot_id'] == 'public-' + digest({k:v for k,v in archive.items() if k != 'snapshot_id'})[:16], 'Public archive payload hash mismatch')
    for name, url in PUBLIC_FILES.items():
        source = archive['source_files'][name]
        require(source['url'] == url and bool(re.fullmatch('[0-9a-f]{64}', source['sha256'])) and source['retrieved_at'], 'Public archive provenance missing')
    coverage = coverage_audit(archive['weekly'], archive['start'], archive['end'])
    require(coverage == archive['coverage'], 'Public coverage audit differs')
    years = defaultdict(list)
    for row in archive['annual']:
        require(row.get('title') and row.get('artist'), 'Annual identity missing')
        years[row['year']].append(row['rank'])
    require(set(years) == set(range(2006, 2022)), 'Annual cohort missing years')
    for ranks in years.values():
        require(len(ranks) == 100 and set(ranks) == set(range(1,101)), 'Incomplete annual evidence')
    return {**coverage, 'archive_hash': digest(archive), 'snapshot_id': archive['snapshot_id'], 'source_authority_independently_authenticated': False}


def public_label_all(records, archive, config):
    from datetime import date
    from .schema import candidate, protocol
    from . import LABEL_VERSION
    protocol(config)
    require(config.get('cohort') == COHORT and config.get('evidence_policy') == 'public_song_query_v1', 'Public song evidence requires explicit academic cohort config')
    require(config['chart_cutoff'] == archive['end'], 'Public cutoff mismatch')
    require(len({r['recording_id'] for r in records}) == len(records), 'Duplicate candidate ID')
    canonical = [r['canonical_recording_id'] for r in records if r.get('canonical_recording_id')]
    require(len(set(canonical)) == len(canonical), 'Duplicate canonical recording')
    validate_public_archive(archive)
    index = ChartIndex(archive['weekly'], archive['annual'], archive['coverage'])
    archive_hash = digest(archive)
    # Original labels are exclusion/candidate constraints, never proof of a new label.
    import pandas as pd
    from chorus_hit.config import ROOT
    legacy = pd.read_csv(ROOT / 'data/chorus_features.csv', usecols=['track_id', 'title', 'artist', 'label']).to_dict('records')
    by_id = {x['track_id']: x for x in legacy}
    result = []
    for r in sorted(records, key=lambda x: x['recording_id']):
        require(r.get('cohort') == COHORT and r.get('synthetic') is False, 'Public metadata candidate cohort/provenance mismatch')
        candidate(r)
        row = {**r, 'label': None, 'label_status': 'unknown', 'label_rule_version': LABEL_VERSION,
               'cohort': COHORT, 'chart_name': 'Billboard Hot 100', 'chart_region': 'US',
               'archive_cutoff_date': config['chart_cutoff'], 'chart_coverage_snapshot_id': archive['snapshot_id'],
               'chart_coverage_complete': False, 'chart_entry_dates': [], 'chart_entry_evidence_urls': [],
               'negative_absence_evidence': None, 'excluded_reason': None}
        result.append(row)
        if r['identity_status'] != 'verified':
            row['excluded_reason'] = 'unresolved_metadata_identity'; continue
        release, cutoff = iso(r['first_release_date']), iso(config['chart_cutoff'])
        anniversary = date(release.year + 2, release.month, min(release.day, 28)) if release.month == 2 and release.day == 29 else release.replace(year=release.year + 2)
        if anniversary > cutoff:
            row['excluded_reason'] = 'insufficient_24_month_followup'; continue
        if release < iso(archive['start']):
            row['excluded_reason'] = 'release_before_audited_window'; continue
        titles, artists = r.get('chart_title_aliases', [r['title']]), r.get('chart_artist_aliases', [r['original_artist_credit']])
        require(isinstance(titles, list) and titles and isinstance(artists, list) and artists, 'Chart aliases required')
        if any(alternate_version_title(t) for t in titles):
            row['excluded_reason'] = 'alternate_version_requires_review'; continue
        old = by_id.get(r.get('legacy_track_id'))
        if r.get('legacy_track_id'):
            require(old and old['label'] == 1 and normalized(old['title']) in {normalized(t) for t in titles}
                    and credit_matches(old['artist'], artists), 'Year-end positive must reuse a matching original class-1 candidate')
        if any(x['label'] == 0 and normalized(x['title']) in {normalized(t) for t in titles} and credit_matches(x['artist'], artists) for x in legacy):
            row['excluded_reason'] = 'original_weekly_class_excluded'; continue
        hits = index.hits(titles, artists, config['chart_cutoff']); annual = index.yearend(titles, artists)
        row['chart_observation_start'] = r['first_release_date']; row['chart_observation_end'] = config['chart_cutoff']
        row['chart_coverage_complete'] = set(weeks_between(r['first_release_date'], config['chart_cutoff'])) <= set(archive['coverage']['covered_weeks'])
        if any(h['date'] < r['first_release_date'] for h in hits):
            row['excluded_reason'] = 'chart_predates_recording_release_review_required'; continue
        if annual and hits and r.get('legacy_track_id'):
            row.update(label=1, label_status='verified_year_end_hit', annual_evidence=annual,
                       chart_entry_dates=sorted({h['date'] for h in hits}),
                       chart_entry_evidence_urls=[archive['source_files']['weekly_current.csv']['url']])
        elif annual or hits:
            row['excluded_reason'] = 'weekly_charted_or_nonreused_year_end_not_in_initial_cohort'
        else:
            absence = index.absence(titles, artists, r['first_release_date'], config['chart_cutoff'])
            if absence['status'] == 'verified_noncharted_in_public_archive':
                row.update(label=0, label_status='verified_noncharted_in_public_archive',
                           negative_absence_evidence={**absence, 'archive_hash': archive_hash, 'query_rule': 'public_song_query_v1'})
            else:
                row.update(excluded_reason=absence.get('reason', 'unresolved_chart_match'), negative_absence_evidence=absence)
    works = defaultdict(list)
    for row in result:
        if row.get('canonical_work_id'): works[row['canonical_work_id']].append(row)
    for group in works.values():
        if len({r['label'] for r in group if r['label'] is not None}) > 1:
            for row in group:
                row.update(label=None, label_status='unknown', excluded_reason='conflicting_labels_for_related_work')
    return result
