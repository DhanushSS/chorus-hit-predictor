"""Invented fixtures exercise evidence rules; none are training/research results."""
from copy import deepcopy
import json
from pathlib import Path
import pandas as pd
import pytest
from chorus_hit.config import ROOT
from chorus_hit.hit_nonhit.academic import audit_legacy, album_choices, choose_release, recording_row
from chorus_hit.hit_nonhit.academic_evidence import ChartIndex, COHORT, coverage_audit, credit_matches, public_label_all, validate_public_archive
from chorus_hit.hit_nonhit.common import digest, read, sha, write
from chorus_hit.hit_nonhit.public_sources import PublicCache, PUBLIC_FILES, WikiRows, read_public_tables
from chorus_hit.hit_nonhit.sources import weeks_between, validate_archive
from chorus_hit.hit_nonhit.synthetic import inputs
from chorus_hit.hit_nonhit.train import candidate_config
from chorus_hit.hit_nonhit.intake import review


def week(day, change=None):
    rows = [{'date': day, 'rank': i, 'title': f'Invented track {i}', 'artist': f'Invented artist {i}'} for i in range(1,101)]
    if change: rows[0].update(change)
    return rows


def test_coverage_duplicate_title_credit_is_incomplete():
    rows = week('2020-01-04'); rows[1].update(title=rows[0]['title'], artist=rows[0]['artist'])
    result = coverage_audit(rows, '2020-01-04', '2020-01-11')
    assert result['covered_weeks'] == [] and len(result['missing_or_incomplete_weeks']) == 2
    with pytest.raises(ValueError, match='weekday'):
        coverage_audit(week('2020-01-05'), '2020-01-04', '2020-01-11')


def test_credit_boundary_and_aliases():
    assert credit_matches('Beyoncé Featuring Jay-Z', ['Beyonce'])
    assert not credit_matches('Adele Tribute Band', ['Adele'])
    assert not credit_matches('Adele & Other Artist', ['Adele'])  # ambiguous joint credit


@pytest.mark.parametrize('entry,status', [({'title':'Secret Song','artist':'Another Artist'}, 'verified_noncharted_in_public_archive'),
    ({'title':'Secret Song (Remix)','artist':'Artist A'}, 'unknown'),
    ({'title':'Secret Song','artist':'Artist A & Guest'}, 'unknown'),
    ({'title':'Secret Song','artist':'Artist A'}, 'charted')])
def test_public_absence_never_uses_title_alone(entry, status):
    rows = week('2020-01-04', entry)
    idx = ChartIndex(rows, [], coverage_audit(rows, '2020-01-04','2020-01-04'))
    assert idx.absence(['Secret Song'], ['Artist A'], '2020-01-01', '2020-01-04')['status'] == status
    assert idx.absence(['Different'], ['Artist A'], '2019-12-01', '2020-01-04')['status'] == 'unknown'


def test_legacy_weekly_rows_are_never_negative():
    frame = pd.DataFrame([{'track_id':'a','title':'Song','artist':'Singer','label':1}, {'track_id':'b','title':'Weekly','artist':'Singer','label':0}])
    idx = ChartIndex([{'date':'2020-01-04','title':'Song','artist':'Singer'}], [{'year':2020,'rank':1,'title':'Song','artist':'Singer'}], {})
    result = audit_legacy(frame, idx)
    assert result[0]['new_label'] == 1
    assert result[1]['new_label'] is None and result[1]['status'] == 'excluded_legacy_weekly_class'


def test_wikipedia_parser_skips_footnotes():
    p = WikiRows(); p.feed('<table><tr><th>No.</th><th>Title</th><th>Artist</th></tr><tr><td>1</td><td>"Title"<sup>[9]</sup></td><td><a>Artist</a></td></tr></table>')
    assert p.rows == [['1','"Title"','Artist']]


def test_public_cache_hash_and_missing_provenance(tmp_path):
    url = next(iter(PUBLIC_FILES.values())); c = PublicCache(tmp_path)
    p = tmp_path / 'file.csv'; p.write_text('example')
    with pytest.raises(ValueError, match='provenance'): c.fetch(url, 'file.csv')
    write(tmp_path / 'file.csv.source.json', {'url':url,'sha256':sha(p)})
    assert c.fetch(url, 'file.csv') == p
    p.write_text('modified')
    with pytest.raises(ValueError, match='hash/URL'): c.fetch(url, 'file.csv')
    with pytest.raises(ValueError, match='hosts'): c.fetch('http://not-allowed.invalid/data')


def test_small_profile_includes_all_families_and_preserves_strict_profile():
    cfg = read(ROOT / 'configs/hit_nonhit_academic.json')
    values = candidate_config(cfg, False, [f'group{i}' for i in range(20)])
    assert {'logistic','linear_svm','rbf_svm','random_forest','extra_trees','hist_boosting','mlp','dummy'} == {v['family'] for v in values}
    assert {v['representation'] for v in values} == {'all518','std74','pca95'}
    cfg.pop('model_profile')
    with pytest.raises(ValueError, match='50 development groups'): candidate_config(cfg, False, range(20))


def test_release_selection_ignores_live_and_undated():
    releases = [{'id':'a','status':'Official','date':'2020','country':'US'}, {'id':'b','status':'Official','date':'2020-01-01','country':'US','disambiguation':'live'}, {'id':'c','status':'Official','date':'2020-01-01','country':'US'}]
    assert choose_release(releases)['id'] == 'c'
    groups = [{'id':'a','primary-type':'Album','secondary-types':['Live'],'first-release-date':'2010-01-01'}, {'id':'b','primary-type':'Album','secondary-types':[],'first-release-date':'2010-01-01'}, {'id':'c','primary-type':'Album','secondary-types':[],'first-release-date':'2015-01-01'}]
    assert album_choices(groups,[2015],1)[0]['id'] == 'c'


def public_fixture():
    rows, _, _ = inputs(1)
    for row in rows:
        row.update(synthetic=False, first_release_date='2019-01-01', album_release_date='2020-01-01', cohort=COHORT,
                   chart_title_aliases=[row['title']], chart_artist_aliases=[row['original_artist_credit']])
    rows[1].update(title='Blinding Lights', original_artist_credit='The Weeknd', chart_title_aliases=['Blinding Lights'], chart_artist_aliases=['The Weeknd'], legacy_track_id='CH0001')
    # Keep the test archive deliberately incomplete: a positive can be proved,
    # while a negative must remain unknown. Complete absence is tested at index level.
    weekly = week('2020-01-04', {'title':'Blinding Lights','artist':'The Weeknd'})
    annual = [{'year':y,'rank':rank,'title':'Blinding Lights' if rank==1 else f'Annual fixture {rank}','artist':'The Weeknd' if rank==1 else 'Fixture Artist'} for y in range(2006,2022) for rank in range(1,101)]
    archive = {'resolution_policy':'public_song_query_v1','cohort':COHORT,'chart_name':'Billboard Hot 100','chart_region':'US','synthetic':False,
               'start':'2000-01-01','end':'2023-12-30','weekly':weekly,'annual':annual,
               'source_files':{n:{'url':u,'sha256':'a'*64,'retrieved_at':'test fixture only'} for n,u in PUBLIC_FILES.items()},
               'coverage':coverage_audit(weekly,'2000-01-01','2023-12-30')}
    archive['snapshot_id'] = 'public-' + digest(archive)[:16]
    return rows, archive, read(ROOT / 'configs/hit_nonhit_academic.json')


def rehash(a):
    a['coverage'] = coverage_audit(a['weekly'],a['start'],a['end'])
    a['snapshot_id'] = 'public-' + digest({k:v for k,v in a.items() if k != 'snapshot_id'})[:16]


def test_public_archive_dispatch_hash_and_positive_evidence():
    rows,a,cfg=public_fixture()
    assert len(validate_archive(a)['covered_weeks']) == 1
    result = public_label_all(rows,a,cfg)
    assert result[0]['label'] is None and result[1]['label'] == 1
    a['weekly'][0]['title']='tampered'
    with pytest.raises(ValueError,match='payload hash'): validate_public_archive(a)


def test_public_label_does_not_promote_weekly_only_or_unknown_identity():
    rows,a,cfg=public_fixture(); a['annual']=[{**r,'title':'Different title'} for r in a['annual']]; rehash(a)
    result=public_label_all(rows,a,cfg)
    assert result[1]['label'] is None and 'weekly_charted' in result[1]['excluded_reason']
    rows[1]['identity_status']='pending'
    assert public_label_all(rows,a,cfg)[1]['excluded_reason']=='unresolved_metadata_identity'


def test_public_followup_and_legacy_positive_constraints():
    rows,a,cfg=public_fixture(); rows[1].update(first_release_date='2023-01-01',album_release_date='2023-01-01')
    assert public_label_all(rows,a,cfg)[1]['excluded_reason']=='insufficient_24_month_followup'
    rows,a,cfg=public_fixture(); rows[1]['legacy_track_id']='CH9999'
    with pytest.raises(ValueError,match='reuse'): public_label_all(rows,a,cfg)
    rows,a,cfg=public_fixture(); cfg.pop('cohort')
    with pytest.raises(ValueError,match='cohort'): public_label_all(rows,a,cfg)


def test_intake_dry_review_needs_real_files_and_does_not_trust_edited_labels(tmp_path):
    rows,a,cfg=public_fixture(); rid=rows[1]['recording_id']
    fields={'recording_id':rid,'proposed_label':'1','audio_path':'','permission_basis':'','permission_reference':'','reviewer':'','recording_matches':'','chorus_start_seconds':'','chorus_confirmed':''}
    path=tmp_path/'intake.csv'; pd.DataFrame([fields]).to_csv(path,index=False)
    ready, report=review(path,rows,a,cfg,tmp_path)
    assert not ready and report['missing_or_invalid'][0]['reason']=='audio_path not supplied'
    fields['proposed_label']='0';pd.DataFrame([fields]).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Edited/unverified'): review(path,rows,a,cfg,tmp_path)


def test_complete_public_window_can_support_negative_and_reject_hidden_hit():
    rows,a,cfg=public_fixture()
    rows[0].update(first_release_date='2021-12-25', album_release_date='2021-12-25')
    a['weekly']=[r for day in weeks_between('2021-12-25','2023-12-30') for r in week(day)]
    rehash(a)
    result=public_label_all(rows,a,cfg)
    assert result[0]['label']==0 and result[0]['negative_absence_evidence']['weeks_queried']==106
    a['weekly'][100].update(title=rows[0]['title'],artist=rows[0]['original_artist_credit']);rehash(a)
    assert public_label_all(rows,a,cfg)[0]['label'] is None
    a['weekly']=a['weekly'][:-100];rehash(a)
    assert public_label_all(rows,a,cfg)[0]['label'] is None


def test_small_groups_skip_complex_candidates_without_blocking_baselines():
    cfg=read(ROOT/'configs/hit_nonhit_academic.json')
    values=candidate_config(cfg,False,range(10))
    assert values and all(x['family'] not in {'mlp','hist_boosting'} for x in values)
    assert {'logistic','random_forest','extra_trees','linear_svm','rbf_svm'} <= {x['family'] for x in values}


def test_public_prerelease_chart_match_requires_review():
    rows,a,cfg=public_fixture();rows[1].update(first_release_date='2020-01-05',album_release_date='2020-01-05')
    assert public_label_all(rows,a,cfg)[1]['excluded_reason']=='chart_predates_recording_release_review_required'


def test_pilot_is_balanced_round_robin_and_does_not_reuse_controls():
    from chorus_hit.hit_nonhit.academic import select_pilot
    labeled=[]
    for artist in ['A','B','C']:
        for number in range(3):
            hit={'label':1,'artist':artist,'album_id':artist,'recording_id':f'{artist}-hit-{number}', 'canonical_work_id':f'{artist}-hit-work-{number}'}
            control={'label':0,'artist':artist,'album_id':artist,'recording_id':f'{artist}-control-{number}','canonical_work_id':f'{artist}-control-work-{number}','matched_positive_ids':[hit['recording_id']]}
            labeled.extend([hit,control])
    result=select_pilot(labeled,6)
    assert len(result)==6 and sum(r['label'] for r in result)==3
    assert {r['artist'] for r in result}=={'A','B','C'}
    assert len({r['recording_id'] for r in result})==6
    assert result==select_pilot(list(reversed(labeled)),6)


def test_public_table_parser_handles_string_ranks(tmp_path):
    pd.DataFrame([{'chart_week':'2020-01-04','current_week':1,'title':'A','performer':'B'}]).to_csv(tmp_path/'weekly_current.csv',index=False)
    annual=[{'year':year,'rank':str(rank),'title':'A','artist':'B'} for year in range(2006,2021) for rank in range(1,101)]
    pd.DataFrame(annual).to_csv(tmp_path/'yearend.csv',index=False,encoding='cp1252')
    page='<table>'+''.join(f'<tr><td>{rank}</td><td>A</td><td>B</td></tr>' for rank in range(1,101))+'</table>'
    (tmp_path/'yearend_2021.html').write_text(page)
    weekly, result=read_public_tables(tmp_path)
    assert len(result)==1600 and weekly.columns.tolist()==['date','rank','title','artist']
    (tmp_path/'yearend_2021.html').write_text('<html>No data</html>')
    with pytest.raises(ValueError,match='no usable ranking'): read_public_tables(tmp_path)


def test_intake_checks_permissions_version_root_and_chorus(tmp_path):
    rows,a,cfg=public_fixture();rid=rows[1]['recording_id']
    fields={'recording_id':rid,'proposed_label':'1','audio_path':'example.wav','permission_basis':'test generated waveform',
            'permission_reference':'test fixture only','reviewer':'test','recording_matches':'yes','chorus_start_seconds':'','chorus_confirmed':''}
    path=tmp_path/'intake.csv'
    # Existence is sufficient at intake; the decoder validates actual audio during build.
    (tmp_path/'example.wav').write_bytes(b'placeholder for input review test')
    pd.DataFrame([fields]).to_csv(path,index=False)
    ready,report=review(path,rows,a,cfg,tmp_path)
    assert len(ready)==1 and ready[0]['audio_sha256']==sha(tmp_path/'example.wav')
    for key,value,reason in [('permission_basis','','Missing'),('recording_matches','no','named studio'),('audio_path','../outside.wav','escapes'),('chorus_confirmed','yes','start time'),
                             ('chorus_start_seconds','nan','finite'),('chorus_start_seconds','inf','finite'),('chorus_start_seconds','-1','nonnegative')]:
        pd.DataFrame([{**fields,key:value}]).to_csv(path,index=False)
        ready,report=review(path,rows,a,cfg,tmp_path)
        assert not ready and reason in report['missing_or_invalid'][0]['reason']


def test_recording_metadata_never_promotes_dj_mix_or_unknown_work():
    base={'id':'r','title':'Song','first-release-date':'2010-01-01','disambiguation':'',
          'artist-credit':[{'name':'Artist','artist':{'id':'a'}}], 'relations':[{'type':'performance','target-type':'work','work':{'id':'w'}}]}
    track={'id':'t','title':'Song','position':1,'recording':{'title':'Song'}}
    album={'title':'Album','date':'2010-01-02'};group={'id':'g'};artist={'id':'a','name':'Artist'}
    args=[track,album,group,artist,['Artist'],[{'url':'fixture://metadata'}],None]
    assert recording_row(base,*args)['identity_status']=='verified'
    assert recording_row({**base,'disambiguation':'part of a DJ-mix'},*args)['identity_status']=='pending'
    assert recording_row({**base,'relations':[]},*args)['identity_status']=='pending'
    assert recording_row({**base,'first-release-date':'2010'},*args)['identity_status']=='pending'


def test_non_latin_identities_do_not_collapse_to_empty_strings():
    from chorus_hit.hit_nonhit.academic_evidence import normalized
    assert normalized('봄날') and normalized('봄날') != normalized('가을')
    assert not credit_matches('防弾少年団', ['방탄소년단'])


def test_public_metadata_route_builds_both_classes_with_shared_audio_extractor(tmp_path):
    import numpy as np
    import soundfile as sf
    from chorus_hit.hit_nonhit.build_dataset import build, load_dataset
    rows,a,cfg=public_fixture()
    rows[0].update(first_release_date='2021-12-25',album_release_date='2021-12-25')
    a['weekly']=[r for day in weeks_between('2021-12-25','2023-12-30') for r in week(day)]
    a['weekly'][0].update(title='Blinding Lights',artist='The Weeknd');rehash(a)
    for number,row in enumerate(rows):
        t=np.arange(16*22050)/22050
        signal=.3*np.sin(2*np.pi*(220+number*40)*t)+.15*np.sin(2*np.pi*(330+number*70)*t)
        path=tmp_path/row['audio_path'];sf.write(path,signal,22050,subtype='FLOAT')
        row['audio_sha256']=sha(path)
    result=build(a,rows,cfg,tmp_path,tmp_path/'dataset')
    assert result['eligible_audio']==2 and not result['audio_failures']
    X,records,manifest,config=load_dataset(tmp_path/'dataset')
    assert X.shape==(2,519) and {r['label'] for r in records}=={0,1}
    assert all(r['segment_samples']==330750 for r in records)
    assert config['cohort']==COHORT and manifest['synthetic'] is False


def test_academic_prediction_keeps_year_end_target_name():
    import numpy as np
    from chorus_hit.config import FEATURE_COLUMNS
    from chorus_hit.hit_nonhit.artifacts import predict_features
    from chorus_hit.hit_nonhit.audio import contract
    class Model:
        def predict(self, X): return np.array([1])
        def predict_proba(self, X): return np.array([[.3,.7]])
    bundle={'task_id':'hit_nonhit_v1','cohort':COHORT,'extractor':contract(),'estimator':Model(),'chart_cutoff':'2023-12-30'}
    out=predict_features(bundle,pd.DataFrame([[0.]*518],columns=FEATURE_COLUMNS),contract())
    assert out['labels']==['Year-end Hot 100 hit'] and out['cohort']==COHORT
    assert out['score_semantics']['calibrated'] is False


def test_metadata_preflight_preserves_group_support_gates():
    from chorus_hit.hit_nonhit.academic import metadata_preflight
    cfg=read(ROOT/'configs/hit_nonhit_academic.json')
    rows=[]
    for group in range(30):
        for label in [0,1]:
            rows.append({'recording_id':f'g{group}-{label}','canonical_artist_ids':[f'artist{group}'],
                         'canonical_work_id':f'work{group}-{label}','album_id':f'album{group}','label':label})
    result=metadata_preflight(rows,cfg)
    assert result['feasible'] and result['connected_groups']==30
    assert result['heldout']['groups']==6
    assert not metadata_preflight(rows[:8],cfg)['feasible']
    assert 'No test was frozen' in result['scope']


def test_published_metadata_is_balanced_preserved_and_not_audio_results():
    root=ROOT/'data/hit_nonhit_v1';progress=read(root/'academic_progress.json')
    for name,expected in progress['files'].items():
        assert sha(root/name)==expected
    intake=pd.read_csv(root/'audio_intake.csv',keep_default_na=False)
    audit=pd.read_csv(root/'academic/label_audit.csv',keep_default_na=False).set_index('recording_id')
    assert intake.recording_id.is_unique and len(intake)==progress['pilot_songs']
    assert intake.proposed_label.value_counts().to_dict()=={0:139,1:139}
    assert all(audit.loc[r.recording_id,'label']==str(float(r.proposed_label)) for r in intake.itertuples())
    assert (intake.audio_path=='').all() and progress['genuine_feature_rows']==0
    assert progress['new_target_accuracy'] is None and progress['new_target_balanced_accuracy'] is None
    assert progress['old_weekly_examples_excluded']==385
    source=read(root/'academic/source_manifest.json')
    assert source['legacy_sha256']==sha(ROOT/'data/chorus_features.csv')


@pytest.mark.parametrize('title,expected', [('Peacock (Cory Enemy & Mia Moretti vocal club mix)',True), ('Runaway (5.1 mix)',True), ('Song - Live at Wembley',True), ('Live Your Life',False), ('Over My Head (Cable Car)',False)])
def test_alternate_versions_in_titles_are_reviewed(title,expected):
    from chorus_hit.hit_nonhit.academic_evidence import alternate_version_title
    assert alternate_version_title(title) is expected


def test_public_label_excludes_mix_even_with_empty_disambiguation():
    rows,a,cfg=public_fixture();rows[0]['chart_title_aliases']=['Peacock (club mix)']
    result=public_label_all(rows,a,cfg)
    assert result[0]['label'] is None and result[0]['excluded_reason']=='alternate_version_requires_review'


def test_pilot_never_acquires_same_work_twice():
    from chorus_hit.hit_nonhit.academic import select_pilot
    hits=[{'label':1,'artist':'A','album_id':'album','recording_id':f'h{i}','canonical_work_id':f'hwork{i}','legacy_track_id':f'CH{i}'} for i in range(2)]
    controls=[{'label':0,'artist':'A','album_id':'album','recording_id':f'c{i}','canonical_work_id':'samework','matched_positive_ids':['h0','h1']} for i in range(2)]
    result=select_pilot(hits+controls)
    assert len(result)==2 and len({r['canonical_work_id'] for r in result})==2


def test_corrupt_optional_progress_does_not_break_application(tmp_path,monkeypatch):
    from chorus_hit.hit_nonhit import ui
    from streamlit.testing.v1 import AppTest
    folder=tmp_path/'data/hit_nonhit_v1';folder.mkdir(parents=True)
    (folder/'academic_progress.json').write_text('{broken json')
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    app=AppTest.from_string('from chorus_hit.hit_nonhit.ui import render\nimport streamlit as st\nrender(st)').run()
    assert not app.exception
    assert any('progress is unavailable' in x.value for x in app.info)
    assert any('Dataset/model not ready' in x.value for x in app.info)
