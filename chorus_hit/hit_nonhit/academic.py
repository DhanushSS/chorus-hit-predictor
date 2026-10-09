"""Practical public-metadata preparation. Downloads metadata only, never recordings."""
import argparse
from collections import Counter
from pathlib import Path
import re
import pandas as pd
from chorus_hit.config import ROOT
from . import TASK
from .academic_evidence import COHORT, ChartIndex, coverage_audit, normalized, credit_matches, alternate_version_title
from .common import cli, digest, read, sha, stage, stamp, write
from .public_sources import PublicCache, PUBLIC_FILES, read_public_tables
from .schema import iso, require

CUTOFF = '2023-12-30'
START = '2000-01-01'
VERSION_WORDS = re.compile(r'\b(live|remix|mix|demo|karaoke|instrumental|acoustic|interview|rehearsal)\b', re.I)
INTAKE_FIELDS = ['recording_id', 'title', 'artist', 'album', 'proposed_label', 'legacy_track_id',
                 'audio_path', 'permission_basis', 'permission_reference', 'reviewer',
                 'recording_matches', 'chorus_start_seconds', 'chorus_confirmed']


def audit_legacy(frame, index):
    rows = []
    for r in frame.to_dict('records'):
        annual = index.yearend([r['title']], [r['artist']])
        weekly = index.hits([r['title']], [r['artist']], CUTOFF)
        status = ('verified_year_end_candidate' if annual and weekly else 'unknown_needs_alias_review') if r['label'] == 1 else 'excluded_legacy_weekly_class'
        rows.append({'legacy_track_id': r['track_id'], 'title': r['title'], 'artist': r['artist'],
                     'legacy_label': r['label'], 'status': status,
                     'year_end_years': sorted({x['year'] for x in annual}),
                     'weekly_first': min((x['date'] for x in weekly), default=None),
                     'weekly_last': max((x['date'] for x in weekly), default=None),
                     'annual_evidence': annual, 'weekly_evidence_count': len(weekly),
                     'new_label': 1 if status == 'verified_year_end_candidate' else None})
    return rows


def prepare(cache_dir, destination):
    c = PublicCache(cache_dir)
    for name, url in PUBLIC_FILES.items():
        c.fetch(url, name)
    weekly, annual = read_public_tables(cache_dir)
    weekly = weekly[weekly.date <= CUTOFF]
    coverage = coverage_audit(weekly.to_dict('records'), START, CUTOFF)
    index = ChartIndex(weekly.to_dict('records'), annual.to_dict('records'), coverage)
    original = ROOT / 'data/chorus_features.csv'
    provenance = read(ROOT / 'data/provenance.json')
    require(sha(original) == provenance['prepared_sha256'], 'Original dataset hash differs from its provenance')
    reused = audit_legacy(pd.read_csv(original)[['track_id', 'title', 'artist', 'label']], index)
    archive = {'resolution_policy': 'public_song_query_v1', 'chart_name': 'Billboard Hot 100', 'chart_region': 'US',
               'synthetic': False, 'cohort': COHORT, 'start': START, 'end': CUTOFF,
               'weekly': weekly.to_dict('records'), 'annual': annual.to_dict('records'),
               'source_files': {n: read(c.directory / (n + '.source.json')) for n in PUBLIC_FILES},
               'coverage': coverage, 'limitations': 'Public compiled metadata, not an authenticated Billboard export; song-level artist/title matching, exact recording must be reviewed against supplied audio.'}
    archive['snapshot_id'] = 'public-' + digest(archive)[:16]
    report = {'cohort': COHORT, 'cutoff': CUTOFF, 'legacy_sha256': sha(original),
              'legacy_songs': len(reused), 'legacy_positive_candidates': sum(x['legacy_label'] == 1 for x in reused),
              'reuse_status': dict(Counter(x['status'] for x in reused)),
              'chart_expected_weeks': coverage['expected_weeks'], 'chart_complete_weeks': len(coverage['covered_weeks']),
              'chart_missing_weeks': coverage['missing_or_incomplete_weeks'], 'annual_entries': len(annual),
              'audio_files_provided': 0, 'new_target_accuracy': None, 'faculty_confirmation': 'submission checkpoint, not development gate'}
    with stage(destination) as out:
        write(out / 'archive.json', archive); write(out / 'legacy_audit.json', reused); write(out / 'summary.json', report)
        table = pd.DataFrame([{**{k:v for k,v in x.items() if k != 'annual_evidence'}, 'year_end_years': ';'.join(map(str,x['year_end_years']))} for x in reused])
        table.to_csv(out / 'legacy_audit.csv', index=False)
    return report


def artist_aliases(a):
    # Keep native/English aliases; unrelated transliterations are unnecessary queries.
    return sorted({a['name'], a.get('sort-name', a['name']), *(v['name'] for v in a.get('aliases', []) if v.get('locale') in {None, 'en'})})


def resolve_artist(client, name):
    data, note = client.mb('artist', query='artist:"' + name.replace('"', '') + '"', limit=25)
    matches = [a for a in data.get('artists', []) if normalized(name) in {normalized(s) for s in artist_aliases(a)} and int(a.get('score', 0)) >= 95]
    require(len(matches) == 1, 'Artist search is ambiguous or has no exact name/alias match')
    return matches[0], note


def album_choices(groups, annual_years, limit):
    eligible = [g for g in groups if g.get('primary-type') == 'Album' and not g.get('secondary-types')
                and re.fullmatch(r'20\d\d-\d\d-\d\d', g.get('first-release-date', ''))
                and '2006-01-01' <= g['first-release-date'] <= '2021-12-30']
    def priority(g):
        year = int(g['first-release-date'][:4])
        supported = sum(0 <= y - year <= 2 for y in annual_years)
        return (-supported, g['first-release-date'], g['id'])
    return sorted(eligible, key=priority)[:limit]


def choose_release(releases):
    eligible = [r for r in releases if r.get('status') == 'Official' and re.fullmatch(r'\d{4}-\d{2}-\d{2}', r.get('date',''))
                and r['date'] <= '2021-12-30' and not VERSION_WORDS.search(r.get('disambiguation', ''))]
    require(eligible, 'No dated official studio album edition available')
    # Earliest US edition, else earliest available edition. No model/audio outcomes used.
    return sorted(eligible, key=lambda r: (r.get('country') != 'US', r['date'], bool(r.get('disambiguation')), r['id']))[0]


def recording_row(detail, track, album, group, artist, aliases, notes, legacy_id):
    artists = [c['artist']['id'] for c in detail.get('artist-credit', []) if 'artist' in c]
    works = [r['work']['id'] for r in detail.get('relations', []) if r.get('type') == 'performance' and r.get('target-type') == 'work']
    date = detail.get('first-release-date', '')
    valid = (bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', date)) and len(set(works)) == 1
             and date <= album['date']
             and artist['id'] in artists and not detail.get('video')
             and not VERSION_WORDS.search(detail.get('disambiguation', ''))
             and not alternate_version_title(detail['title'])
             and normalized(detail['title']) == normalized(track['recording']['title']))
    now = stamp()
    return {'dataset_version': TASK, 'synthetic': False, 'recording_id': 'mb-' + detail['id'],
            'title': detail['title'], 'original_artist_credit': ''.join(x.get('name','') + x.get('joinphrase','') for x in detail.get('artist-credit', [])),
            'artist': artist['name'], 'album': album['title'], 'canonical_artist_ids': sorted(set(artists)),
            'canonical_recording_id': detail['id'], 'canonical_work_id': works[0] if len(set(works)) == 1 else '',
            'album_id': group['id'], 'track_id': track['id'], 'track_number': int(track['position']),
            'recording_version': 'official_album_metadata; supplied_audio_not_yet_reviewed',
            'first_release_date': date, 'album_release_date': album['date'], 'market': 'US',
            'identity_source': 'MusicBrainz core metadata', 'identity_status': 'verified' if valid else 'pending',
            'identity_reviewer': 'deterministic_metadata_checks_v1', 'identity_reviewed_at': now,
            'identity_evidence_urls': sorted({n['url'] for n in notes}), 'metadata_sources': notes,
            'version_policy': 'public_song_metadata_pending_audio_review',
            'source_license_snapshot': 'MusicBrainz core data CC0 https://musicbrainz.org/doc/About/Data_License',
            'created_at': now, 'source_checked_at': now, 'chart_artist_aliases': aliases,
            'chart_title_aliases': sorted({track['title'], track['recording']['title'], detail['title']}),
            'legacy_track_id': legacy_id, 'cohort': COHORT}


def discover(prepared, cache_dir, destination, artist_limit=40, albums_per_artist=3):
    require(1 <= artist_limit <= 80 and 1 <= albums_per_artist <= 4, 'Discovery bounds exceeded')
    archive = read(Path(prepared) / 'archive.json'); audit = read(Path(prepared) / 'legacy_audit.json')
    index = ChartIndex(archive['weekly'], archive['annual'], archive['coverage']); client = PublicCache(cache_dir)
    positives = [r for r in audit if r['legacy_label'] == 1]
    counts = Counter(r['artist'] for r in positives)
    names = sorted(counts, key=lambda n: (-counts[n], normalized(n)))[:artist_limit]
    found, errors, album_log = {}, [], []
    for number, name in enumerate(names, 1):
        print(f'Artist {number}/{len(names)}: {name}', flush=True)
        try:
            artist, artist_note = resolve_artist(client, name)
            aliases = sorted(set(artist_aliases(artist) + [name]))
            legacy = [r for r in positives if r['artist'] == name]
            years = [y for r in legacy for y in r['year_end_years']]
            groups = []
            for offset in range(0, 300, 100):
                data, gn = client.mb('release-group', artist=artist['id'], type='album', limit=100, offset=offset)
                groups.extend(data['release-groups'])
                if len(groups) >= data['release-group-count']: break
            for group in album_choices(groups, years, albums_per_artist):
                try:
                    releases, rn = client.mb('release', **{'release-group': group['id'], 'status':'official', 'limit':100})
                    chosen = choose_release(releases['releases'])
                    album, an = client.mb('release/' + chosen['id'], inc='recordings+artist-credits+release-groups')
                    require(album['release-group']['id'] == group['id'], 'Album group mismatch')
                    tracks = [t for m in album['media'] for t in m.get('tracks', []) if t.get('recording') and not VERSION_WORDS.search(t['recording'].get('disambiguation', ''))]
                    hits, controls = [], []
                    for track in tracks:
                        titles = list({track['title'], track['recording']['title']})
                        old = next((r for r in legacy if normalized(r['title']) in {normalized(t) for t in titles}), None)
                        if old and index.yearend(titles, aliases) and index.hits(titles, aliases, CUTOFF):
                            hits.append((track, old['legacy_track_id']))
                        elif not index.yearend(titles, aliases) and index.absence(titles, aliases, START, CUTOFF)['status'] == 'verified_noncharted_in_public_archive':
                            controls.append(track)
                    # At most two controls per reused positive; final selection is paired/balanced.
                    chosen_controls = sorted(controls, key=lambda t: digest(t['recording']['id']))[:2 * len(hits)]
                    album_log.append({'artist': name, 'album': album['title'], 'release_id': album['id'], 'positive_anchors': len(hits), 'possible_controls': len(controls)})
                    for track, legacy_id in hits + [(t, None) for t in chosen_controls]:
                        rid = track['recording']['id']
                        if rid in found: continue
                        detail, dn = client.mb('recording/' + rid, inc='work-rels+artist-credits')
                        found[rid] = recording_row(detail, track, album, group, artist, aliases, [artist_note, gn, rn, an, dn], legacy_id)
                except (ValueError, OSError, KeyError) as error:
                    errors.append({'artist': name, 'album': group['title'], 'error': str(error)})
        except (ValueError, OSError, KeyError) as error:
            errors.append({'artist': name, 'error': str(error)})
    return curate(list(found.values()), archive, album_log, errors, names, destination)


def select_pilot(labeled, maximum=300):
    """Balance pairs while visiting each artist before taking another of its hits."""
    from collections import defaultdict
    from itertools import zip_longest
    require(type(maximum) is int and 2 <= maximum <= 300 and maximum % 2 == 0, 'Pilot size must be an even number from 2 to 300')
    by_artist = defaultdict(list)
    for r in sorted((r for r in labeled if r['label'] == 1), key=lambda r: (r['artist'], r['album_id'], r['recording_id'])):
        by_artist[r['artist']].append(r)
    selected = []; used_controls = set(); used_works = set(); used_legacy = set()
    for round_hits in zip_longest(*(by_artist[a] for a in sorted(by_artist))):
        for hit in round_hits:
            if hit is None: continue
            if hit.get('canonical_work_id') in used_works or (hit.get('legacy_track_id') and hit['legacy_track_id'] in used_legacy): continue
            options = [r for r in labeled if r['label'] == 0 and hit['recording_id'] in r['matched_positive_ids'] and r['recording_id'] not in used_controls
                       and r.get('canonical_work_id') not in used_works and r.get('canonical_work_id') != hit.get('canonical_work_id')]
            if not options: continue
            control = sorted(options, key=lambda r: (r['album_id'] != hit['album_id'], digest(r['recording_id'])))[0]
            selected.extend([hit, control]); used_controls.add(control['recording_id'])
            used_works.update(r['canonical_work_id'] for r in [hit, control] if r.get('canonical_work_id'))
            if hit.get('legacy_track_id'): used_legacy.add(hit['legacy_track_id'])
            if len(selected) >= maximum: return selected
    return selected


def metadata_preflight(selected, config):
    """Check group support before requesting audio; this does not freeze a test."""
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit
    from .audit import connected_groups
    from .freeze_split import grouped_folds
    groups = np.asarray(connected_groups(selected)); y = np.asarray([r['label'] for r in selected])
    e = config['evaluation']
    report = {'scope': 'Metadata only; audio hashes/signatures and missing recordings may change support. No test was frozen or evaluated.',
              'connected_groups': len(set(groups)), 'feasible': False}
    try:
        require(len(selected) > 0, 'No selected recordings')
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=e['test_fraction'], random_state=e['seed']).split(y, y, groups))
        for name, ix in [('development', train), ('heldout', test)]:
            report[name] = {'rows': len(ix), 'groups': len(set(groups[ix])),
                            'class_counts': {str(k): int(sum(y[ix] == k)) for k in [0, 1]},
                            'groups_per_class': {str(k): len(set(groups[ix][y[ix] == k])) for k in [0, 1]}}
            require(all(n >= max(5, e['minimum_test_groups_per_class']) for n in report[name]['groups_per_class'].values()), 'Insufficient independent class support')
        for inner_train, _ in grouped_folds(y[train], groups[train], e['outer_folds'], e['seed']):
            grouped_folds(y[train][inner_train], groups[train][inner_train], e['inner_folds'], e['seed'])
        report['feasible'] = True
    except ValueError as error:
        report['reason'] = str(error)
    return report


def curate(rows, archive, album_log, errors, names, destination):
    from .academic_evidence import public_label_all
    from .matching import match_candidates
    config = read(ROOT / 'configs/hit_nonhit_academic.json')
    labeled = match_candidates(public_label_all(rows, archive, config))
    selected = select_pilot(labeled)
    records = [{k: ({'artist':r['artist'], 'proposed_label':r['label']}.get(k, r.get(k, ''))) for k in INTAKE_FIELDS} for r in selected]
    report = {'cohort': COHORT, 'discovered_recordings': len(rows), 'metadata_label_status': dict(Counter(r['label_status'] for r in labeled)),
              'matched_eligible': dict(Counter(str(r['label']) for r in labeled if r['label'] is not None and r['match_level'] != 'unmatched')),
              'pilot_songs': len(selected), 'pilot_class_counts': dict(Counter(str(r['label']) for r in selected)),
              'pilot_artists': len({r['artist'] for r in selected}), 'audio_files_provided': 0, 'accuracy': None,
              'metadata_preflight': metadata_preflight(selected, config),
              'unknown_reasons': dict(Counter(r['excluded_reason'] for r in labeled if r['label'] is None)),
              'selection': 'same-album then same-artist-within-two-years controls; deterministic identity-hash tie break; round-robin artists; up to 300 songs, before audio or ML outcomes',
              'errors': errors, 'requested_artists': names}
    with stage(destination) as out:
        write(out / 'candidates.json', rows); write(out / 'labels.json', labeled); write(out / 'summary.json', report)
        write(out / 'albums.json', album_log)
        pd.DataFrame(records, columns=INTAKE_FIELDS).to_csv(out / 'audio_intake.csv', index=False)
    return report


def main():
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--cache', required=True); prep.add_argument('--out', required=True)
    disc = sub.add_parser('discover'); disc.add_argument('--prepared', required=True); disc.add_argument('--cache', required=True); disc.add_argument('--out', required=True)
    disc.add_argument('--artists', type=int, default=40); disc.add_argument('--albums-per-artist', type=int, default=3)
    cur = sub.add_parser('curate'); cur.add_argument('--prepared', required=True); cur.add_argument('--source', required=True); cur.add_argument('--out', required=True)
    args = p.parse_args()
    if args.action == 'prepare': print(prepare(args.cache, args.out))
    elif args.action == 'discover': print(discover(args.prepared, args.cache, args.out, args.artists, args.albums_per_artist))
    else:
        source = Path(args.source); summary = read(source / 'summary.json')
        print(curate(read(source / 'candidates.json'), read(Path(args.prepared) / 'archive.json'), read(source / 'albums.json'), summary['errors'], summary['requested_artists'], args.out))


if __name__ == '__main__': cli(main)
