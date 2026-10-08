"""Predeclared inclusive matching: keep all viable same-artist controls, no score filtering."""
from .common import digest


def match_candidates(rows):
    hits = [r for r in rows if r['label'] == 1]
    result = []
    for row in rows:
        if row['label'] is None:
            result.append({**row, 'match_level': 'unmatched', 'matched_positive_ids': [], 'match_set_id': None,
                           'release_year_delta': None, 'viable_matches': 0, 'selection_rule': 'unresolved_identity_excluded'})
            continue
        artists = set(row['canonical_artist_ids'])
        viable = [h for h in hits if artists & set(h['canonical_artist_ids'])]
        same_album = [h for h in viable if row['album_id'] == h['album_id']]
        era = [h for h in viable if abs(int(row['first_release_date'][:4]) - int(h['first_release_date'][:4])) <= 2]
        matches = same_album or era
        level = 'same_album' if same_album else 'same_artist_era' if era else 'unmatched'
        result.append({**row, 'match_level': level, 'matched_positive_ids': sorted(h['recording_id'] for h in matches),
                       'match_set_id': ('match-' + digest([level, row['album_id'], sorted(artists)])[:16]) if matches else None,
                       'release_year_delta': min([abs(int(row['first_release_date'][:4]) - int(h['first_release_date'][:4])) for h in matches], default=None),
                       'viable_matches': len(matches), 'selection_rule': 'all_same_album_else_same_artist_within_2_years_v1'})
    return result
