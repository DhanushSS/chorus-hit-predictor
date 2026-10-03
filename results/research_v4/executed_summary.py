"""Verify the declared nested comparison and fixed-candidate diagnostic."""
from collections import Counter
from pathlib import Path
import json
import sys
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chorus_hit.artifacts import load_run
from chorus_hit.audit import atomic_json, sha256_file, json_hash
from chorus_hit.data import normalize_artist
from chorus_hit.estimators import make_estimator
from chorus_hit.evaluation import classification_metrics, paired_bootstrap, bootstrap_intervals, score_model


def main():
    out=ROOT/'results/research_v4'
    if (out/'comparison_manifest.json').exists():
        raise FileExistsError('Completed comparison is immutable')
    before,after=load_run('v3_nested_001'),load_run('v4_std74_001')
    config=json.loads((after.path/'config.json').read_text())
    preflight=json.loads((out/'preflight.json').read_text())
    assert sha256_file(ROOT/'configs/v4_std74.json')==preflight['config_sha256']
    assert {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'chorus_hit').glob('*.py'))}==preflight['source_files']
    assert json_hash(preflight['source_files'])==after.manifest['code_sha256']
    a=before.predictions.set_index('track_id').sort_index()
    b=after.predictions.set_index('track_id').loc[a.index]
    assert a.label.equals(b.label) and a.artist_group.equals(b.artist_group)
    folds=json.loads((after.path/'outer_folds.json').read_text())
    assert folds==json.loads((before.path/'outer_folds.json').read_text())
    for stage in [*[f'outer{i}' for i in range(5)],'final_development']:
        assert json.loads((before.path/f'{stage}_folds.json').read_text())==json.loads((after.path/f'{stage}_folds.json').read_text())
    paired=paired_bootstrap(a.label,b.prediction,a.prediction,a.artist_group)
    trials=[json.loads(p.read_text()) for p in (after.path/'trials').glob('*.json')]
    statuses=dict(Counter(t['status'] for t in trials))
    warning_count=sum(len(t['warnings']) for t in trials)+len(after.summary['final_fit_warnings'])
    ranking=pd.read_csv(after.path/'final_development_ranking.csv')
    new=ranking[ranking.candidate_id.str.startswith('v4_')].sort_values('mean_balanced_accuracy',ascending=False)
    old_source=before.data.set_index('track_id').loc[after.bundle['train_track_ids']]
    oldpred,oldscore,_=before.predict(old_source[before.bundle['features']])
    newpred,newscore,_=after.predict(old_source[after.bundle['features']])
    comparison={'run_id':after.summary['run_id'],'comparison_run':before.summary['run_id'],
        'metrics_before':before.summary['metrics'],'metrics_after':after.summary['metrics'],
        'paired_vs_v3':paired,'ci95':after.summary['ci95'],
        'changed_oof_predictions':int((a.prediction!=b.prediction).sum()),
        'additional_correct_oof_predictions':int((b.prediction==b.label).sum()-(a.prediction==a.label).sum()),
        'fold_trial_count':len(trials),'final_refits':1,'trial_statuses':statuses,'warnings':warning_count,
        'final_candidate':after.summary['model_name'],'retained_dimensions':after.summary['retained_dimensions'],
        'same_final_development_predictions':bool(np.array_equal(oldpred,newpred)),
        'same_final_development_scores':bool(np.array_equal(oldscore,newscore)),
        'new_candidate_wins_in_outer_folds':sum(x['candidate_id'].startswith('v4_') for x in json.loads((after.path/'outer_fold_metrics.json').read_text())),
        'best_added_full_development_tuning':new.iloc[0].to_dict(),
        'seconds':after.manifest['elapsed_seconds'],'fresh_test_available':False,'historical_test_evaluated':False,
        'active_run':json.loads((ROOT/'configs/active_run.json').read_text())['run_id'],
        'target_status':after.summary['target_status'],
        'limitations':'Previously inspected development records; fixed-prediction group-bootstrap intervals omit some selection and training uncertainty.'}

    # This diagnostic was fixed in the protocol before the nested study ran.
    diagnostic=out/'fixed_candidate_diagnostic';diagnostic.mkdir(exist_ok=False)
    names=['lr_mi_50','v4_lr_std74_quantile_0.1']
    specs={s['id']:s for s in config['candidates']}
    data=after.data.set_index('track_id')
    dev=set(after.bundle['train_track_ids'])
    records=[];fit_log=[]
    with threadpool_limits(limits=1):
        for f in folds:
            fit_ids,valid_ids=f['fit_track_ids'],f['validation_track_ids']
            assert set(fit_ids)|set(valid_ids)==dev and not set(fit_ids)&set(valid_ids)
            train,val=data.loc[fit_ids],data.loc[valid_ids]
            assert not set(train.artist.map(normalize_artist))&set(val.artist.map(normalize_artist))
            for name in names:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model=make_estimator(specs[name],after.bundle['features'],config['seed'])
                    model.fit(train[after.bundle['features']],train.label)
                    prediction=model.predict(val[after.bundle['features']])
                    score,semantics=score_model(model,val[after.bundle['features']])
                path=diagnostic/f'{name}_fold{f["fold"]}.joblib'
                joblib.dump(model,path,compress=3)
                restored=joblib.load(path)
                assert np.array_equal(prediction,restored.predict(val[after.bundle['features']]))
                if name.startswith('v4_'):
                    selected=model.named_steps['columns'].columns
                    assert len(selected)==74
                    assert np.array_equal(model.named_steps['imputer'].statistics_,train[selected].median().to_numpy())
                    unscaled=model[:3].transform(train[after.bundle['features']])
                    assert np.array_equal(model.named_steps['scale'].quantiles_[-1],unscaled.max(axis=0))
                fit_log.append({'candidate':name,'fold':f['fold'],'status':'complete','warnings':[str(w.message) for w in caught],
                    'metrics':classification_metrics(val.label,prediction,score),'reload_verified':True})
                for i,(track_id,r) in enumerate(val.iterrows()):
                    records.append({'track_id':track_id,'artist_group':normalize_artist(r.artist),'label':int(r.label),
                        'candidate':name,'fold':f['fold'],'prediction':int(prediction[i]),'score':float(score[i])})
    predictions=pd.DataFrame(records);predictions.to_csv(diagnostic/'predictions.csv',index=False)
    diag={}
    for name in names:
        rows=predictions[predictions.candidate==name].set_index('track_id').loc[a.index]
        assert rows.label.equals(a.label)
        diag[name]={'metrics':classification_metrics(rows.label,rows.prediction),
            'ci95':bootstrap_intervals(rows.label,rows.prediction,rows.artist_group)['intervals']}
    p0=predictions[predictions.candidate==names[0]].set_index('track_id').loc[a.index]
    p1=predictions[predictions.candidate==names[1]].set_index('track_id').loc[a.index]
    diag['paired_archive_vs_incumbent']=paired_bootstrap(p0.label,p1.prediction,p0.prediction,p0.artist_group)
    diag['purpose']='Descriptive fixed-candidate comparison on already inspected folds; no ranking or final model selection from these scores.'
    atomic_json(diagnostic/'summary.json',diag);atomic_json(diagnostic/'fit_log.json',fit_log)
    atomic_json(diagnostic/'manifest.json',{'status':'complete','files':{p.name:sha256_file(p) for p in diagnostic.iterdir() if p.is_file()}})
    comparison['fixed_candidate_diagnostic']=diag
    atomic_json(out/'comparison.json',comparison)
    new.to_csv(out/'added_candidates_tuning.csv',index=False)

    rows=[]
    for key in ['accuracy','balanced_accuracy','precision','recall','f1']:
        rows.append({'metric':key,'V3':before.summary['metrics'][key],'V4':after.summary['metrics'][key]})
    table=pd.DataFrame(rows);table.to_csv(out/'metrics_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.6));x=np.arange(len(table))
    ax.bar(x-.18,table.V3*100,.36,label='Previous V3 procedure',color='#8AA9AD')
    ax.bar(x+.18,table.V4*100,.36,label='V4 with 74-feature candidates',color='#147D83')
    ax.axhline(75,color='#AB6D21',ls='--',label='75% boundary')
    ax.set_xticks(x,['Accuracy','Balanced\naccuracy','Precision','Recall','F1']);ax.set_ylim(0,100);ax.set_ylabel('Percent')
    for i,v in enumerate(table.V4*100):ax.text(i+.18,v+1,f'{v:.1f}',ha='center',fontsize=9)
    ax.legend(frameon=False,fontsize=9);ax.set_title('Same 597 songs and five outer artist-grouped folds',loc='left')
    fig.text(.06,.025,'Exploratory development comparison. No fresh test; no historical re-evaluation.',fontsize=9,color='#53666E')
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)

    delta=paired['balanced_accuracy_difference']*100;lo,hi=np.array(paired['ci95'])*100
    fixed_delta=diag['paired_archive_vs_incumbent']
    fixed1=diag[names[1]]['metrics'];fixed0=diag[names[0]]['metrics']
    body=f'''# Archive-inspired 74-feature experiment - 4 October 2026

## Outcome

The declared V4 selection procedure achieved **{after.summary['metrics']['balanced_accuracy']:.2%} balanced accuracy** and **{after.summary['metrics']['accuracy']:.2%} ordinary accuracy**, versus V3's {before.summary['metrics']['balanced_accuracy']:.2%} and {before.summary['metrics']['accuracy']:.2%}. The paired balanced-accuracy change is **{delta:+.2f} percentage points**, with an approximate 95% whole-artist interval of **{lo:+.2f} to {hi:+.2f} points**. {'The interval includes zero; a reliable improvement is not established.' if lo<=0<=hi else 'This conditional interval excludes zero, but reused development records still require fresh confirmation.'}

![Comparison](../results/research_v4/comparison.png)

## Protocol and changes

The task remains source year-end hit versus other charted song. The same 597 development songs and 52 artist-name groups, five outer folds, three inner folds, seeds and native thresholds were retained. All 55 V3 candidates remained eligible. Eight additions used the 74 standard-deviation features, balanced logistic regression, standard or 100-quantile normal scaling, and C in 0.003, 0.01, 0.03, 0.1. The raw input contract remains 518 columns; each new pipeline selects 74 internally. Every learned transform fits only its training fold.

`configs/v4_std74.json`, estimator support and isolation tests were committed at `{preflight['source_commit']}` before execution. The nested run completed {len(trials)} recorded fold fits plus one final refit in {after.manifest['elapsed_seconds']:.2f} seconds, with {statuses.get('failed',0)} failed fold fits and {warning_count} fit warnings. The separate, predeclared fixed-candidate diagnostic completed ten fits. Search selection uses inner-fold balanced accuracy, not outer diagnostic scores.

| Metric | Previous V3 | V4 expanded procedure |
| --- | ---: | ---: |
'''
    body+='\n'.join(f"| {r.metric.replace('_',' ').title()} | {r.V3:.2%} | {r.V4:.2%} |" for r in table.itertuples())
    body+=f'''

V4 balanced-accuracy interval: {after.summary['ci95']['balanced_accuracy'][0]:.2%}-{after.summary['ci95']['balanced_accuracy'][1]:.2%}. Changed outer predictions: {comparison['changed_oof_predictions']}; net additional correct predictions: {comparison['additional_correct_oof_predictions']}. A new 74-feature candidate won {comparison['new_candidate_wins_in_outer_folds']} of five inner selections. Final full-development selection: **{after.summary['model_name']}**, retaining {after.summary['retained_dimensions']} features. Final development predictions equal the previous fitted model: {comparison['same_final_development_predictions']}; scores equal: {comparison['same_final_development_scores']}.

## Fixed candidate diagnostic

On the same five outer folds, the archive's C=0.1/quantile-100 model scored **{fixed1['balanced_accuracy']:.2%} balanced accuracy** and {fixed1['accuracy']:.2%} ordinary accuracy; the frozen incumbent scored **{fixed0['balanced_accuracy']:.2%} balanced accuracy** and {fixed0['accuracy']:.2%} ordinary accuracy. The paired BA difference is {100*fixed_delta['balanced_accuracy_difference']:+.2f} points, interval {100*fixed_delta['ci95'][0]:+.2f} to {100*fixed_delta['ci95'][1]:+.2f}. These are descriptive fixed-model results, distinct from the nested selection procedure. They were not used to choose the final model. All ten fitted models reproduce saved predictions after reload; training-only feature/imputation/quantile boundaries were checked.

## Interpretation and limits

The archive's 57.13% was a selection-CV mean, not a nested or new-test estimate. The highest added full-development tuning candidate here is `{new.iloc[0].candidate_id}` at {new.iloc[0].mean_balanced_accuracy:.2%}; this also is a tuning score. No favourable seeds, thresholds, labels or songs were selected after seeing these results. The five-metric strict >75% target is {'met in this development evaluation' if after.summary['target_status']['target_met_in_this_evaluation'] else 'not met'}.

These data and earlier archive results have already been inspected. Bootstrap intervals condition on saved predictions and do not capture all model-selection/training uncertainty. Artist-name identity and chart labels remain incompletely verified; original matching recordings and a genuinely new evaluation collection are missing. No new historical evaluation or fresh-test claim is made. The active demo remains `{comparison['active_run']}`; V4 is an optional research run.

## Reproduction and status

- Implemented and run: 74-feature pipelines, bounded nested search, paired comparison, fixed-candidate diagnostic, reload and input-boundary checks.
- Blocked: matched-audio representations and independent confirmation need verified matching recordings, credited-performer/recording identities, chart evidence and new evaluation songs.
- Not attempted: changing the target, new historical scoring, automatic model promotion, or publishing this update.
- Commands: `.venv/bin/python -m chorus_hit.train_v2 --config configs/v4_std74.json --run-id v4_std74_001`; `.venv/bin/python scripts/summarize_std74.py`; `.venv/bin/python -m pytest -q`.
- The completed run refuses overwriting; reproduce in a disposable checkout with this run absent or choose a new run ID and adapt the comparison script. Pinned dependencies are unchanged. Final test evidence is in `results/research_v4/final_tests.log`.
- Evidence: `results/v2/v4_std74_001/`, `results/research_v4/comparison.json`, fixed-candidate predictions/models, training log, executed-source archive and preservation hashes. Current local branch: `research/std74-evaluation`.

Next action: use this result to decide whether this representation merits testing on verified new data. Prioritize recording/label verification and matched permitted audio; another search on the same songs cannot supply independent confirmation.
'''
    (ROOT/'docs/V4_STD74_RESULTS.md').write_text(body)
    preservation=json.loads((out/'preservation.json').read_text())
    assert all(sha256_file(ROOT/name)==digest for name,digest in preservation.items())
    atomic_json(out/'comparison_manifest.json',{'status':'complete','protected_files_unchanged':len(preservation),
        'source_run_manifests':{x.summary['run_id']:sha256_file(x.path/'manifest.json') for x in [before,after]},
        'summary_script_sha256':sha256_file(Path(__file__)),
        'files':{str(p.relative_to(out)):sha256_file(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    main()
