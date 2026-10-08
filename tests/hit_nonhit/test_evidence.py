from copy import deepcopy
import pytest
from chorus_hit.hit_nonhit.common import digest, write
from chorus_hit.hit_nonhit.chart_membership import classify, label_all
from chorus_hit.hit_nonhit.identity import normalize, append_decision, review_queue
from chorus_hit.hit_nonhit.matching import match_candidates
from chorus_hit.hit_nonhit.sources import validate_archive
from chorus_hit.hit_nonhit.synthetic import inputs


@pytest.fixture
def evidence():
    return inputs(2)


def rehash(archive):
    for w in archive['weeks']:
        w['entries_sha256'] = digest(w['entries'])
    archive['source']['export_sha256'] = digest(archive['weeks'])


def test_positive_and_proven_absence(evidence):
    rows, archive, cfg = evidence
    out = label_all(rows, archive, cfg)
    assert [r['label'] for r in out] == [0, 1, 0, 1]
    assert out[0]['negative_absence_evidence']['weeks_sha256'] == digest(out[0]['negative_absence_evidence']['weeks'])
    assert out[1]['chart_entry_evidence_urls']
    matched = match_candidates(out)
    assert matched[0]['match_level'] == 'same_album' and matched[0]['viable_matches'] == 1


def test_missing_one_week_blocks_negative_but_not_positive(evidence):
    rows, archive, cfg = evidence
    hidden = archive['weeks'].pop(15)['date']; rehash(archive)
    assert hidden in validate_archive(archive)['missing_or_incomplete_weeks']
    assert classify(rows[0], archive, cfg)['label'] is None
    assert classify(rows[1], archive, cfg)['label'] == 1


def test_partial_chart_99_rows_not_complete(evidence):
    rows, archive, cfg = evidence
    archive['weeks'][0]['entries'].pop(); rehash(archive)
    assert classify(rows[0], archive, cfg)['excluded_reason'] == 'incomplete_chart_coverage'


def test_hidden_weekly_hit_contradicts_negative(evidence):
    rows, archive, cfg = evidence
    e = archive['weeks'][17]['entries'][80]
    e.update(canonical_work_id=rows[0]['canonical_work_id'], canonical_recording_ids=[rows[0]['canonical_recording_id']], canonical_artist_ids=rows[0]['canonical_artist_ids'])
    rehash(archive)
    result = classify(rows[0], archive, cfg)
    assert result['label'] == 1 and result['chart_entry_dates'] == [archive['weeks'][17]['date']]


def test_same_title_different_artist_never_matches(evidence):
    rows, archive, cfg = evidence
    rows[0]['title'] = rows[1]['title']
    assert classify(rows[0], archive, cfg)['label'] == 0


@pytest.mark.parametrize('change,reason', [('work','unresolved_chart_identity_or_related_version'),('pending','unresolved_identity_or_version'),('release','insufficient_24_month_followup')])
def test_ambiguous_versions_and_followup(evidence, change, reason):
    rows, archive, cfg = evidence
    if change == 'work': rows[0]['canonical_work_id'] = rows[1]['canonical_work_id']
    if change == 'pending': rows[0]['identity_status'] = 'ambiguous'
    if change == 'release': rows[0].update(first_release_date='2021-01-01', album_release_date='2021-01-01')
    assert classify(rows[0], archive, cfg)['excluded_reason'] == reason


def test_any_unresolved_chart_entry_prevents_absence(evidence):
    rows, archive, cfg = evidence
    archive['weeks'][0]['entries'][99]['identity_status'] = 'unresolved'; rehash(archive)
    assert classify(rows[0], archive, cfg)['label'] is None


def test_post_cutoff_hit_does_not_relabel(evidence):
    rows, archive, cfg = evidence
    week = deepcopy(archive['weeks'][-1]); week['date'] = '2022-01-15'
    week['entries'][99].update(canonical_recording_ids=[rows[0]['canonical_recording_id']], canonical_work_id=rows[0]['canonical_work_id'], canonical_artist_ids=rows[0]['canonical_artist_ids'])
    archive['end'] = week['date']; archive['weeks'].append(week); rehash(archive)
    assert classify(rows[0], archive, cfg)['label'] == 0


@pytest.mark.parametrize('change', ['region','duplicate_week','wrong_date','rank','hash','rights','identity_collision'])
def test_bad_sources_fail_closed(evidence, change):
    rows, a, cfg = evidence
    if change == 'region': a['chart_region'] = 'UK'
    if change == 'duplicate_week': a['weeks'].append(deepcopy(a['weeks'][0])); rehash(a)
    if change == 'wrong_date': a['weeks'][0]['date'] = '2020-01-05'; rehash(a)
    if change == 'rank': a['weeks'][0]['entries'][0]['rank'] = 100; rehash(a)
    if change == 'hash': a['weeks'][0]['entries'][0]['title'] = 'tampered'
    if change == 'rights': a['source']['research_use_authorized'] = False
    if change == 'identity_collision': a['weeks'][0]['entries'][0]['canonical_recording_ids'] = a['weeks'][0]['entries'][1]['canonical_recording_ids']; rehash(a)
    with pytest.raises(ValueError): validate_archive(a)


def test_no_cutoff_or_legacy_relabel(evidence):
    rows, a, cfg = evidence
    cfg['chart_cutoff'] = None
    with pytest.raises(ValueError, match='cutoff'): classify(rows[0], a, cfg)
    cfg['chart_cutoff'] = '2022-01-08'; rows[0]['dataset_version'] = 'legacy'
    with pytest.raises(ValueError, match='legacy'): classify(rows[0], a, cfg)


def test_canonical_collision_and_prerelease_are_unknown(evidence):
    rows, a, cfg = evidence
    rows[1]['canonical_work_id'] = 'different_work'
    assert classify(rows[1], a, cfg)['excluded_reason'] == 'canonical_identity_collision'
    rows[1]['canonical_work_id'] = a['weeks'][0]['entries'][0]['canonical_work_id']
    rows[1].update(first_release_date='2020-01-05', album_release_date='2020-01-05')
    assert classify(rows[1], a, cfg)['excluded_reason'] == 'chart_predates_first_release_review_required'


def test_aliases_are_only_review_hints_and_log_is_append_only(tmp_path):
    assert normalize('BEYONCÉ — Live!') == normalize('Beyoncé Live')
    decision = {'recording_id':'r', 'reviewer':'a', 'reviewed_at':'2026-10-08', 'evidence_urls':['synthetic://test'], 'resolution':'unresolved', 'reason':'version uncertain'}
    one = append_decision(tmp_path, decision); two = append_decision(tmp_path, decision)
    assert two['previous_sha256'] == digest(one) and len(list(tmp_path.glob('*.json'))) == 2
    with pytest.raises(FileExistsError): write(next(tmp_path.glob('*.json')), {})


def test_unresolved_candidate_needs_no_invented_identity_or_release(evidence):
    rows, archive, cfg = evidence
    pending = {'dataset_version':'hit_nonhit_v1','recording_id':'UNKNOWN_A','title':'Unresolved song','original_artist_credit':'Unresolved credit','identity_status':'pending','synthetic':True}
    other = {**pending,'recording_id':'UNKNOWN_B'}
    labeled = label_all(rows + [pending, other],archive,cfg)
    matched = match_candidates(labeled)
    unknown = [r for r in matched if r['label'] is None]
    assert len(unknown) == 2 and all(r['chart_observation_start'] is None for r in unknown)
    assert len(review_queue(unknown)) == 2
    assert sum(r['label'] == 0 for r in matched) == 2
