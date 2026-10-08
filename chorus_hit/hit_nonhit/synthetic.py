"""Invented identities, charts and features for software testing ONLY. Not study evidence."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from chorus_hit.config import FEATURE_COLUMNS, ROOT
from . import TASK, LABEL_VERSION
from .audio import contract
from .audit import summary
from .chart_membership import label_all
from .common import code_identity, digest, environment, read, seal, stage, write
from .matching import match_candidates
from .sources import validate_archive, weeks_between


def inputs(groups=30):
    cfg = read(ROOT / 'configs/hit_nonhit_v1.json')
    cfg.update(chart_cutoff='2022-01-08', real_data_training_status='synthetic_test_only', minimum_reviewed_per_class=0)
    cfg['evaluation'].update(outer_folds=3, inner_folds=2, minimum_test_groups_per_class=1)
    cfg['candidates'] = [cfg['candidates'][0], next(c for c in cfg['candidates'] if c['family'] == 'logistic' and c['representation'] == 'std74')]
    cfg['permutation_repeats'] = 1
    rows = []
    for i in range(groups):
        for label in [0, 1]:
            rid = f'SYNTH_{i:02d}_{label}'
            rows.append({'dataset_version': TASK, 'recording_id': rid, 'canonical_recording_id': rid,
                         'canonical_work_id': 'work_' + rid, 'canonical_artist_ids': [f'artist_{i:02d}'],
                         'album_id': f'album_{i:02d}', 'track_id': rid, 'title': f'Invented tone {rid}',
                         'original_artist_credit': f'Invented artist {i}', 'recording_version': 'studio', 'track_number': label + 1,
                         'first_release_date': '2020-01-01', 'album_release_date': '2020-01-01', 'market': 'US',
                         'identity_source': 'synthetic fixture', 'identity_status': 'verified', 'identity_reviewer': 'fixture',
                         'identity_reviewed_at': '2022-01-09T00:00:00Z', 'identity_evidence_urls': ['synthetic://fixture/identity'],
                         'version_policy': 'exact_recording_reviewed', 'source_license_snapshot': 'synthetic-CC0',
                         'created_at': '2022-01-09T00:00:00Z', 'source_checked_at': '2022-01-09T00:00:00Z',
                         'synthetic': True, 'audio_source_type': 'synthetic_tone', 'audio_rights_basis': 'generated fixture',
                         'audio_processing_authorized': True, 'audio_permission_evidence': 'synthetic://fixture',
                         'audio_rights_reviewer': 'fixture', 'audio_path': rid + '.wav', 'audio_sha256': digest(rid),
                         'duplicate_review_status': 'reviewed', 'duplicate_review_evidence': 'synthetic unique test identities'})
    entries = []
    for rank in range(1, 101):
        r = rows[(rank - 1) * 2 + 1] if rank <= groups else None
        rid = r['canonical_recording_id'] if r else f'FILLER_{rank}'
        entries.append({'rank': rank, 'title': r['title'] if r else rid, 'artist_credit': 'Synthetic artist',
                        'identity_status': 'verified', 'canonical_work_id': r['canonical_work_id'] if r else 'work_' + rid,
                        'canonical_recording_ids': [rid], 'canonical_artist_ids': r['canonical_artist_ids'] if r else ['artist_' + rid],
                        'identity_evidence': 'synthetic://fixture/chart', 'identity_reviewer': 'fixture', 'version_reviewed': True})
    weeks = [{'date': d, 'entries': deepcopy(entries), 'entries_sha256': digest(entries), 'evidence_url': 'synthetic://fixture/' + d, 'reviewer': 'fixture'} for d in weeks_between('2020-01-01', cfg['chart_cutoff'])]
    archive = {'synthetic': True, 'snapshot_id': 'SYNTHETIC_NOT_REAL_CHARTS', 'chart_name': 'Billboard Hot 100', 'chart_region': 'US',
               'start': '2020-01-01', 'end': cfg['chart_cutoff'], 'weeks': weeks,
               'source': {'provider': 'invented unit test data', 'license': 'CC0', 'license_snapshot': 'synthetic-CC0',
                          'permission_evidence': 'generated locally for tests', 'research_use_authorized': True,
                          'reviewer': 'fixture', 'checked_at': '2022-01-09', 'export_sha256': digest(weeks), 'synthetic': True}}
    return rows, archive, cfg


def feature_package(destination, groups=30):
    candidates, archive, cfg = inputs(groups)
    rows = match_candidates(label_all(candidates, archive, cfg))
    rng = np.random.default_rng(123)
    values = rng.normal(size=(len(rows), len(FEATURE_COLUMNS)))  # no real music, no real accuracy claim
    for r, values_row in zip(rows, values):
        r.update(extractor_version=contract()['extractor_version'], extractor_config_hash=digest(contract()),
                 segment_samples=330750, sample_rate=22050, segment_start_seconds=0., segment_length_seconds=15.,
                 features_schema_hash=digest(FEATURE_COLUMNS), feature_row_hash=digest(values_row.tolist()),
                 selection_method='fallback_excerpt', chorus_annotation_status='unverified', audio_original_format='WAV',
                 recording_duration=20., original_sample_rate=22050, original_channels=1)
    with stage(destination) as tmp:
        table = pd.DataFrame(values, columns=FEATURE_COLUMNS); table.insert(0, 'recording_id', [r['recording_id'] for r in rows])
        table.to_csv(tmp / 'features.csv', index=False, float_format='%.17g')
        for name, obj in [('records', rows), ('candidates', candidates), ('candidate_labels', rows), ('archive', archive), ('config', cfg), ('dataset_audit', summary(rows, rows))]:
            write(tmp / (name + '.json'), obj)
        seal(tmp, {'kind': 'dataset', 'synthetic': True, 'label_definition_version': LABEL_VERSION,
                   'chart_cutoff': cfg['chart_cutoff'], 'extractor': contract(), 'dataset_hash': digest(rows),
                   'coverage': validate_archive(archive), 'pinned_env': environment(), 'code': code_identity()})
    return Path(destination)
