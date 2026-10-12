"""Publish aggregate development evidence; never load the historical test labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chorus_hit.hsp_s.data import development
from chorus_hit.hsp_s.modeling import metrics


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(n):return f'{n:.2%}'
def interval(r):return '–'.join(pct(x) for x in r['balanced_accuracy'])


def publish(run,split,out):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    r=json.loads((run/'results.json').read_text());assert r['status']=='DEVELOPMENT_ONLY'
    frame,columns,_,_=development(split)
    assert not r['identity']['old_test_loaded'] and r['identity']['rows']==len(frame)
    (out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    predictions=json.loads((run/'development_predictions.json').read_text())
    assert predictions['uuid']==frame.uuid.tolist()
    p=np.array(predictions['nested']);y=frame.label.to_numpy();g=frame.group.to_numpy();X=frame[columns]
    rows=[]
    for label,part in frame.groupby('label'):
        rows.append([int(label),len(part),int(part.group.nunique()),int(part[columns].isna().sum().sum()),
                     round(float(part.nuisance_duration.median()),2),round(float(part.nuisance_duration.quantile(.1)),2),round(float(part.nuisance_duration.quantile(.9)),2)])
    source_rows=[]
    for codec,part in frame.groupby('nuisance_codec'):
        mask=frame.nuisance_codec.to_numpy()==codec
        source_rows.append([codec,len(part),int((part.label==0).sum()),int((part.label==1).sum()),part.group.nunique(),pct(float(np.mean(y[mask]==p[mask])))])
    sample_rates=[{'sample_rate':str(k),'class_counts':{str(c):int((v.label==c).sum()) for c in [0,1]},'groups':v.group.nunique()} for k,v in frame.groupby('nuisance_sample_rate')]
    corr=X.corr().to_numpy();high=int(np.sum(np.triu(np.abs(corr)>.98,k=1)))
    group_errors=[{'rows':int(mask.sum()),'accuracy':float((y[mask]==p[mask]).mean())} for group in np.unique(g) for mask in [g==group]]
    desc={'scope':'Post-run descriptive diagnostics; never used for selection or cohort exclusion','sample_rates':sample_rates,'high_absolute_pearson_pairs_over_0p98':high,'group_accuracy_quantiles':{str(q):float(np.quantile([v['accuracy'] for v in group_errors],q)) for q in [0,.25,.5,.75,1]},'groups_with_any_oof_error':sum(v['accuracy']<1 for v in group_errors)}
    (out/'descriptive_diagnostics.json').write_text(json.dumps(desc,indent=2)+'\n')
    d=r['diagnostics']; t=r['trials'];warn=[v for v in t if v.get('warnings')];fail=[v for v in t if v['status']!='complete']
    comparisons=[[c['candidate']['id'],pct(c['balanced_accuracy']) if c['balanced_accuracy'] is not None else 'INELIGIBLE'] for c in r['final_comparison']]
    metric_rows=[[k.replace('_',' ').title(),pct(r['fixed_baseline'][k]),pct(r['nested'][k])] for k in ['accuracy','balanced_accuracy','precision','recall','f1','macro_f1']]
    confusion=r['nested']['confusion_matrix']
    report=f'''# Development-only model improvement report

## Decision

**{r['target_status']}. New untouched external test: NOT AVAILABLE.**
The existing packaged SVM and historical 73.07% accuracy / 73.17% balanced accuracy
are preserved. No new model is deployed or given an independent-test claim.

## Data and design

HSP-S, Vötter et al. (2021), [CC BY 4.0 source](https://zenodo.org/records/5383858):
7,736 original records; eight documented duplicate/conflict exclusions; 7,728
accepted (3,865 chart-entry / 3,863 publisher no-entry). Only the frozen **6,235
song / 2,106 connected-group development set**, class counts
`{r['identity']['class_counts']}`, enters new fits. All 436 inputs are whole-recording
scalar audio descriptors; no chart, popularity or artist metadata predictors.
Negative labels do not establish lifetime non-hit status. No new audio acquired.

Config [research_v2.json](../../configs/research_v2.json) was committed before
execution (`4d028c9`). Seed 20261008; 5 outer / 3 inner group folds; 11 candidates;
90 seconds per fit; 7,200 seconds accumulated fit budget. All imputation,
variance filtering, scaling and supervised feature selection fit inside each
training fold. Candidate selection uses inner mean balanced accuracy, not outer
scores. Diagnostic controls were fixed in advance; no seed or label changes.

## Nested development results

{table(['Metric','Fixed C=1 baseline, same folds','Nested candidate-selection procedure'],metric_rows)}

BA 95% connected-group percentile bootstrap intervals (1,000 draws):
fixed baseline **{interval(r['fixed_baseline_interval'])}**; nested procedure
**{interval(r['nested_interval'])}**. Intervals are conditional on this cohort and
OOF predictions; they do not capture repeated study-design selection uncertainty.
The original 27-candidate nested development BA was **70.94%**; its estimate is
not identical to evaluating a single fixed baseline. These comparisons do not
prove a statistically reliable improvement or independent generalization.

Nested OOF confusion matrix, rows true [no-entry, entry], columns predicted same:
`{confusion}`. Every development record receives one outer-fold prediction.

![Development model comparison](model_comparison.png)

## All registered configurations

{table(['Configuration','Final development 3-fold selection BA'],comparisons)}

These final-selection CV values rank candidates; they are **not nested estimates
for each candidate** or external scores. Outer-fold selections:
`{[f['chosen']['id'] for f in r['outer_folds']]}`.
Final selection: `{json.dumps(r['selected_candidate'],sort_keys=True)}`.
SVMs use balanced classes, max_iter=20000 and cache_size=256. HistGradientBoosting:
200 iterations, 15 leaves, learning_rate=.08, L2=5, no internal random early-stop
split. ExtraTrees: 300 trees, min_samples_leaf=3, balanced, one thread.
Hard voting: SVM + HistGradientBoosting + balanced LogisticRegression C=.1,
max_iter=3000; no in-sample stacking or fabricated probabilities.

The bounded grid covers robust/quantile scaling, ANOVA 75% feature selection,
C/gamma changes and stronger learner families. Threshold tuning, probabilities,
external boosting libraries, additional family-subset/PCA grids and stacking
were not added after inspecting scores. This is a finite study, not a claim of
exhaustive optimization or maximum attainable accuracy.

## Warnings, attempts and reproducibility

Completed run: **{len(t)} trials; {len(fail)} failed; {len(warn)} warned; 0 skipped**.
Recorded accumulated trial wall time: **{sum(x['seconds'] for x in t):.1f} seconds**;
bootstrap/report time is additional. Full trial outcomes, timings, warning text,
outer selections, intervals and config/data/code hashes are in [results.json](results.json).
Warnings count by category: `{dict(Counter(str(w) for item in warn for w in item['warnings']))}`.
No warning was removed, no failed candidate silently accepted. [Run log](run_log.md)
records the interrupted preliminary run and logging correction; its completed
trial records are preserved in [aborted_run_001.json](aborted_run_001.json).
The final implementation was committed as `fdc0e1f` before run 002.

```bash
python -m chorus_hit.research_v2.run --split output/hsp_s/split_001 --out output/research_v2/reproduction_001
python scripts/summarize_research_v2.py --split output/hsp_s/split_001 --run output/research_v2/reproduction_001 --out output/research_v2/reproduction_report
```

Use a **new destination** and the immutable split prepared by the original
[HSP-S protocol](../hit_nonhit/hsp_s_protocol.md). A new clone can download the
pinned source and run `prepare` into a fresh split location with the original
config; do not run the test evaluator. The development loader verifies file
hashes and cross-partition identity groups but only deserializes development
features/labels. Hashing the old locked file for integrity is not a new test
measurement. The access ledger, dataset, bundle and historical results are unchanged.

## Bias, sensitivity and explainability

See [group_leakage_audit.md](group_leakage_audit.md) for all diagnostics.
Codec/source confounding remains unresolved. The historical nuisance-only BA of
61.08% and within-group permutations around 66% remain valid cautions; neither
was erased. Global-null controls answer a different question and do not certify
absence of artist or collection bias. No future-popularity or MP3-accuracy claim
follows from these descriptor-only development results.
'''
    (out/'model_improvement_report.md').write_text(report)
    audit=f'''# Group, source and leakage audit — development only

Existing identity graph validation confirms disjoint known connected groups
across frozen partitions. The new research reads only development labels;
metadata identifiers are used for grouping/diagnostics, never as predictors.
Unknown artist aliases, covers, alternate masters and metadata omissions remain
unverifiable without independent review; passing the graph check does not prove
they are absent. The original held-out identity coverage is 527 groups,
from the immutable split receipt; no new held-out error analysis was performed.

## Group and feature checks

- Development: 2,106 groups / 6,235 rows; size quantiles `{d['group_size_quantiles']}`.
- Pure-label groups: {d['pure_label_groups']}; mixed-label groups: {d['mixed_label_groups']}.
- Duplicate exact development feature vectors: {d['duplicate_feature_rows']}.
- Missing cells: {d['missing_cells']}; constant features: {d['constant_features']}.
- Descriptive feature pairs with absolute Pearson correlation >.98: {high}.
  This was not used for post-hoc feature removal.
- Groups with at least one nested OOF error: {desc['groups_with_any_oof_error']}.
  Group-accuracy quantiles: `{desc['group_accuracy_quantiles']}`. Many small
  groups have discrete 0/1 accuracy, so these are descriptive, not reliable
  individual-artist performance estimates.

{table(['Class','Rows','Groups','Missing cells','Median duration (s)','Duration p10','Duration p90'],rows)}

## Codec and source distribution

{table(['Codec','Rows','Class 0','Class 1','Groups','Nested OOF accuracy'],source_rows)}

Per-codec scores are post-run descriptive slices, not separately selected models;
very small codecs cannot support reliable estimates. All development rows report
extractor `{d['extractor']}`. [Sample-rate counts and descriptive metadata](descriptive_diagnostics.json)
are also published. No genre or reliable release-era fields exist in the frozen
metadata; no guessed positional join or label-derived era was substituted.

Predeclared baseline controls, each with 3-fold grouped OOF evaluation:

{table(['Subset','Rows','Groups','Accuracy','BA','BA 95% group interval'],[[name,str(d[key]['rows']),str(d[key]['groups']),pct(d[key]['metrics']['accuracy']),pct(d[key]['metrics']['balanced_accuracy']),interval(d[key]['interval'])] for name,key in [('MP3 codec only','mp3_only'),('Without largest connected group','without_largest_group')]])}

These restrict codec / remove the single metadata-defined largest component;
they do **not** create a fully matched collection. Fold/cohort composition differs
from the full 5-fold experiment. Samples were not removed based on prediction
errors and these controls did not determine the deployed model. A causal source-
invariant representation and genre/era-matched benchmark remain unavailable.

## Permutation interpretation

Global shuffled-label BA (three predeclared permutations):
**{', '.join(pct(v) for v in d['global_permutation_balanced_accuracy'])}**.
These destroy overall descriptor/label association while retaining the group
splits. They are sanity controls, not calibrated exchangeable-group hypothesis
tests or a p-value. The original within-group permutations retain group label
structure, since pure-label groups cannot change. Their ~66% scores therefore
do not contradict global-null behavior or establish a software bug. They warn
that class-pure identities and cohort/source structure matter.

## Feature-family association (not causation)

Across three grouped validation folds, mean BA drops after jointly permuting
one family across validation rows:

{table(['Family','Mean BA drop (percentage points)'],[[fam,f"{100*np.mean([x[fam] for x in d['feature_family_permutation_drop']]):.2f}"] for fam in ['lowlevel.','rhythm.','tonal.']])}

This perturbation can break correlations between families. It is model
sensitivity, not causal songwriting advice, a calibrated probability, or an
individual-song explanation. See [raw diagnostics](results.json).
'''
    (out/'group_leakage_audit.md').write_text(audit)
    labels=[v[0] for v in comparisons];values=[100*v['balanced_accuracy'] for v in r['final_comparison']]
    fig,ax=plt.subplots(figsize=(9,5.8)); ax.barh(labels,values,color='#227c70');ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Balanced accuracy (%) · selection CV, not independent test');ax.set_title('11 registered configurations · development only')
    for i,v in enumerate(values):ax.text(v+.5,i,f'{v:.2f}%',va='center',fontsize=9)
    fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ['run','split','out']:p.add_argument('--'+n,required=True)
    a=p.parse_args();publish(a.run,a.split,a.out)
