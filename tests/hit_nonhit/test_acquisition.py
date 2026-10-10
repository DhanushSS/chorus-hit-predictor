"""Acquisition scope/privacy regressions; no network, copyrighted audio or research scores."""
from pathlib import Path
import subprocess

import pandas as pd
import pytest
import requests

from chorus_hit.config import ROOT
from chorus_hit.hit_nonhit import acquisition as a
from chorus_hit.hit_nonhit.common import read
from chorus_hit.hit_nonhit.intake import ingest


def test_fixed_cohort_accepts_subsets_but_rejects_expansion_relabel_and_duplicates():
    table = a.selected_cohort().iloc[:2].copy()
    a.validate_selected(table)
    for changed, message in [
        (table.assign(recording_id=['outside', table.iloc[1].recording_id]), 'outside fixed'),
        (table.assign(proposed_label=['0', '0']), 'Edited fixed'),
        (pd.concat([table, table]), 'Duplicate'),
    ]:
        with pytest.raises(ValueError, match=message):
            a.validate_selected(changed)


def test_ingest_scope_guard_runs_before_archive_or_audio_access(tmp_path):
    file = tmp_path / 'intake.csv'
    pd.DataFrame([{'recording_id': 'outside', 'proposed_label': '1'}]).to_csv(file, index=False)
    with pytest.raises(ValueError, match='outside fixed'):
        ingest('missing archive', 'missing candidates', file, 'missing audio', dry_run=True)


def test_changed_committed_cohort_is_not_silently_adopted(tmp_path, monkeypatch):
    file = tmp_path / 'changed.csv'
    file.write_text('recording_id,proposed_label\nnew,1\n')
    monkeypatch.setattr(a, 'COHORT', file)
    with pytest.raises(ValueError, match='cohort changed'):
        a.selected_cohort()


def test_local_ledger_does_not_invent_audio_or_review_and_never_overwrites(tmp_path):
    out = tmp_path / 'ledger'
    summary = a.initialize(out)
    ledger = pd.read_csv(out / 'acquisition_ledger_278.csv', keep_default_na=False)
    assert len(ledger) == 278 and ledger.recording_id.is_unique
    assert list(ledger.columns) == a.LEDGER_FIELDS
    assert summary['metadata_class_counts'] == {'0': 139, '1': 139}
    assert summary['ledger_files_supplied'] == 0
    for field in ['candidate_source_url', 'local_audio_path', 'sha256', 'permission_basis', 'permission_reference']:
        assert (ledger[field] == '').all()
    assert (ledger.disposition == 'MISSING').all()
    assert (ledger.download_permitted == 'unknown').all()
    assert (ledger.analysis_permitted == 'unknown').all()
    assert (ledger.chorus_review_status == 'unreviewed').all()
    assert (ledger.match_set_id != '').all()
    with pytest.raises(ValueError, match='already exists'):
        a.initialize(out)


def test_count_response_missing_and_merged_ids_are_distinct():
    assert a.parse_counts({'x': {'count': 3}}, ['x', 'y']) == {'x': 3, 'y': 0}
    assert a.parse_counts({'x': {'count': 3}, 'mbid_mapping': {'x': 'z'}}, ['x']) == {'x': None}


@pytest.mark.parametrize('payload', [[], {'error': 'unavailable'}, {'x': {'count': -1}},
                                    {'x': {'count': True}}, {'x': {'count': '4'}}, {'x': 4}])
def test_malformed_count_payload_is_never_a_missing_or_positive_result(payload):
    with pytest.raises(ValueError):
        a.parse_counts(payload, ['x'])


def test_feature_probe_errors_stay_unknown_and_429_stops_requests(tmp_path):
    class Response:
        status_code = 429
        headers = {}
        def raise_for_status(self):
            raise requests.HTTPError('rate limited')
    class Client:
        headers = {}
        calls = 0
        def get(self, url, params, timeout):
            assert url == 'https://acousticbrainz.org/api/v1/count'
            assert len(params['recording_ids'].split(';')) <= 25
            self.calls += 1
            return Response()
    client = Client()
    summary = a.probe_features(tmp_path / 'probe', session=client, sleep=lambda _: None)
    assert client.calls == 1
    assert summary['status_counts'] == {'UNKNOWN': 278}
    assert summary['compatible_518_chorus_rows'] == summary['audio_files_downloaded'] == 0
    assert len(read(tmp_path / 'probe/coverage.json')) == 278


def test_feature_probe_counts_are_not_reported_as_chorus_vectors(tmp_path):
    class Response:
        status_code = 200
        headers = {}
        def __init__(self, ids): self.ids = ids
        def raise_for_status(self): pass
        def json(self): return {i: {'count': 1} for i in self.ids}
    class Client:
        headers = {}
        def get(self, url, params, timeout): return Response(params['recording_ids'].split(';'))
    summary = a.probe_features(tmp_path / 'probe', session=Client(), sleep=lambda _: None)
    assert summary['status_counts'] == {'FEATURES_PRESENT': 278}
    assert summary['features_present_class_counts'] == {'0': 139, '1': 139}
    assert summary['compatible_518_chorus_rows'] == summary['audio_files_downloaded'] == 0


def test_private_intakes_and_raw_audio_are_ignored():
    names = ['audio_intake_filled_278.csv', 'acquisition_ledger_278.csv', 'song.aiff', 'song.opus',
             'output/hit_nonhit/recording.wav', 'results/hit_nonhit/new/model.joblib']
    result = subprocess.run(['git', 'check-ignore', '--stdin'], cwd=ROOT, input='\n'.join(names),
                            text=True, capture_output=True, check=True)
    assert set(result.stdout.splitlines()) == set(names)
