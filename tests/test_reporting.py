import copy,json
from chorus_hit.artifacts import load_run
from chorus_hit.reporting import summary_text
from chorus_hit.evaluation import target_status,classification_metrics
from chorus_hit.evaluate_run import load_historical
from chorus_hit.config import ROOT


def test_model_schema_and_conclusion_are_dynamic():
    s=copy.deepcopy(load_run('v2_nested_001').summary)
    s.update(model_name='Fixture Extra Trees',feature_count=4)
    s['metrics'].update(accuracy=.85,balanced_accuracy=.82,precision=.81,recall=.84,f1=.83)
    s['ci95']['balanced_accuracy']=[.78,.87];s['target_status']=target_status(s['metrics'],s['evaluation_status'])
    text=' '.join(summary_text(s).values())
    assert 'Fixture Extra Trees' in text and '4 ordered' in text and '82.0%' in text
    assert 'lies above 50%' in text and 'target is met' in text
    assert '518' not in text and '46.6%' not in text and 'includes 50%' not in text


def test_run_predictions_recompute_and_original_membership():
    a=load_run('v2_nested_001');pred=a.predictions
    assert set(pred.track_id)==set(a.bundle['train_track_ids'])
    assert pred.track_id.nunique()==597
    outer=json.loads((a.path/'outer_folds.json').read_text())
    byid=a.data.set_index('track_id')
    for fold in outer:
        assert set(fold['fit_track_ids']).isdisjoint(fold['validation_track_ids'])
        assert set(byid.loc[fold['fit_track_ids']].artist).isdisjoint(byid.loc[fold['validation_track_ids']].artist)
    assert classification_metrics(pred.label,pred.prediction)['balanced_accuracy']==a.summary['metrics']['balanced_accuracy']
    h=load_historical(ROOT/'results/evaluations/v2_nested_001_historical')
    assert h['target_status']['fresh_test_target_met'] is None


def test_app_run_switch_uses_matching_artifacts():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    app.sidebar.selectbox[0].set_value('v2_nested_001').run(timeout=60)
    assert not app.exception
    assert any('lr_mi_50' in c.value for c in app.caption)
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert app.success or app.warning
