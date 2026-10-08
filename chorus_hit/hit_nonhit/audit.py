"""Connected identity/duplicate groups, evidence counts and cohort/source diagnostics."""
from collections import Counter
from datetime import date
import numpy as np
from .common import digest
from .schema import require


def connected_groups(rows):
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i, j):
        parents[root(j)] = root(i)
    owner = {}
    for i, r in enumerate(rows):
        tokens = [('artist', a) for a in r['canonical_artist_ids']]
        tokens += [(k, r.get(k)) for k in ['canonical_work_id', 'canonical_recording_id', 'album_id', 'match_set_id',
                                            'dedupe_group_id', 'audio_sha256', 'feature_row_hash', 'perceptual_duplicate_group'] if r.get(k)]
        tokens += [('recording_id', r['recording_id'])] + [('recording_id', x) for x in r.get('matched_positive_ids', [])]
        for token in tokens:
            if token in owner:
                union(i, owner[token])
            else:
                owner[token] = i
    # Reviewers can supply robust fingerprint clusters. This additional conservative
    # candidate grouping can reduce support but can never loosen identity isolation.
    for i, r in enumerate(rows):
        if 'perceptual_signature' not in r:
            continue
        for j in range(i):
            if 'perceptual_signature' in rows[j] and np.dot(r['perceptual_signature'], rows[j]['perceptual_signature']) >= .995:
                union(i, j)
    members = {}
    for i, r in enumerate(rows):
        members.setdefault(root(i), []).append(r['recording_id'])
    return ['group-' + digest(sorted(members[root(i)]))[:20] for i in range(len(rows))]


def assert_isolated(rows, left, right):
    require(not set(left) & set(right), 'Record membership overlaps')
    ids = [r['recording_id'] for r in rows]
    groups = dict(zip(ids, connected_groups(rows)))
    require(not {groups[x] for x in left} & {groups[x] for x in right}, 'Connected artist/work/album/match/audio leakage')


def summary(rows, eligible):
    out = {'candidates': len(rows), 'positive': sum(r['label'] == 1 for r in rows),
           'negative': sum(r['label'] == 0 for r in rows), 'unknown': sum(r['label'] is None for r in rows),
           'unknown_reasons': dict(Counter(r['excluded_reason'] for r in rows if r['label'] is None)),
           'eligible_audio': len(eligible), 'artists': len({a for r in eligible for a in r['canonical_artist_ids']}),
           'albums': len({r['album_id'] for r in eligible}), 'works': len({r['canonical_work_id'] for r in eligible}),
           'recordings': len({r['canonical_recording_id'] for r in eligible}),
           'match_sets': len({r['match_set_id'] for r in eligible if r['match_set_id']}),
           'match_levels': dict(Counter(r['match_level'] for r in eligible)),
           'selection_methods': dict(Counter(r['selection_method'] for r in eligible)),
           'real_chorus_detector_accuracy': None}
    out['reviewed_chorus_by_class'] = {str(y): sum(r['label'] == y and r['chorus_annotation_status'] == 'reviewed' for r in eligible) for y in [0, 1]}
    out['observation_days_by_class'] = {str(y): [( date.fromisoformat(r['archive_cutoff_date']) - date.fromisoformat(r['first_release_date'])).days for r in eligible if r['label'] == y] for y in [0, 1]}
    out['release_year_counts_by_class'] = {str(y): dict(Counter(r['first_release_date'][:4] for r in eligible if r['label'] == y)) for y in [0, 1]}
    out['source_by_class'] = {str(y): dict(Counter(r['audio_source_type'] for r in eligible if r['label'] == y)) for y in [0, 1]}
    return out
