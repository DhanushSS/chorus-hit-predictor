"""Recompute the declared exploratory comparison and render its research note."""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import sys
import zipfile

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chorus_hit.artifacts import load_run
from chorus_hit.audit import json_hash, sha256_file
from chorus_hit.evaluation import paired_bootstrap, TARGET_METRICS


def main():
    out = ROOT / 'results/research_v3'
    before, after = load_run('v2_nested_001'), load_run('v3_nested_001')
    p0 = before.predictions.set_index('track_id').sort_index()
    p1 = after.predictions.set_index('track_id').loc[p0.index]
    assert p0.label.equals(p1.label) and p0.artist_group.equals(p1.artist_group)
    assert json.loads((before.path / 'outer_folds.json').read_text()) == json.loads((after.path / 'outer_folds.json').read_text())
    comparison = json.loads((out / 'paired_comparison.json').read_text())
    comparison['paired_vs_v2'] = paired_bootstrap(p0.label, p1.prediction, p0.prediction, p0.artist_group)
    comparison['same_serialized_pipeline_bytes'] = comparison.pop('same_final_pipeline', comparison.get('same_serialized_pipeline_bytes'))
    comparison['changed_oof_predictions'] = int((p0.prediction != p1.prediction).sum())
    comparison['additional_correct_oof_predictions'] = int((p1.prediction == p1.label).sum() - (p0.prediction == p0.label).sum())
    comparison['manifest_sha256'] = {a.summary['run_id']: sha256_file(a.path / 'manifest.json') for a in [before, after]}
    (out / 'paired_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')

    trials = [json.loads(p.read_text()) for p in (after.path / 'trials').glob('*.json')]
    execution = json.loads((out / 'training_command.json').read_text())
    status = dict(Counter(t['status'] for t in trials))
    warning_count = sum(len(t.get('warnings', [])) for t in trials)
    rows = []
    for key in TARGET_METRICS:
        rows.append({'metric': key, 'V2': before.summary['metrics'][key], 'V3': after.summary['metrics'][key],
                     'V3_ci_low': after.summary['ci95'][key][0], 'V3_ci_high': after.summary['ci95'][key][1],
                     'above_75_percent': after.summary['target_status']['metrics'][key]['passed']})
    table = pd.DataFrame(rows)
    table.to_csv(out / 'metrics_comparison.csv', index=False)
    delta = comparison['paired_vs_v2']
    fig, (ax, effect) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})
    x = np.arange(len(rows))
    ax.bar(x - .18, table.V2 * 100, .36, label='V2 nested procedure', color='#92afb4')
    ax.bar(x + .18, table.V3 * 100, .36, label='V3 exploratory procedure', color='#167d87')
    ax.axhline(75, color='#a36016', ls='--', lw=1.5, label='75% boundary')
    ax.set(ylim=(0, 100), ylabel='Percent', title='Same 597 development songs and outer folds')
    ax.set_xticks(x, ['Accuracy', 'Balanced\naccuracy', 'Precision', 'Recall', 'F1'])
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    for i, value in enumerate(table.V3 * 100):
        ax.text(i + .18, value + 1.5, f'{value:.1f}', ha='center', fontsize=8)
    value = delta['balanced_accuracy_difference'] * 100
    lo, hi = np.array(delta['ci95']) * 100
    effect.errorbar([value], [0], xerr=[[value - lo], [hi - value]], fmt='o', color='#167d87', capsize=6)
    effect.axvline(0, color='#52666e', ls='--', lw=1)
    effect.set(yticks=[], ylim=(-1, 1), xlim=(-5, 5), xlabel='Balanced-accuracy change\n(percentage points)', title='Paired 95% interval includes zero')
    effect.text(0, -.55, f'{value:+.2f} points\n95% interval {lo:+.2f} to {hi:+.2f}', ha='center', fontsize=11,
                bbox={'facecolor':'white','edgecolor':'none','pad':4})
    fig.suptitle('V3 robustness study: a small, inconclusive change', fontsize=16, x=.04, ha='left')
    fig.text(.04, .02, 'Whole-artist bootstrap; fixed predictions. Previously inspected development data; no fresh confirmation.', fontsize=9, color='#52666e')
    fig.tight_layout(rect=[0, .10, 1, .92])
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)

    source_files = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted((ROOT / 'chorus_hit').glob('*.py'))}
    assert json_hash(source_files) == after.manifest['code_sha256']
    with zipfile.ZipFile(out / 'executed_code.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for relative in source_files:
            z.write(ROOT / relative, relative)
    evidence = {'run_id': after.summary['run_id'], 'trial_count': len(trials), 'trial_statuses': status,
                'trial_warning_count': warning_count, 'training_command': execution,
                'executed_source_commit': execution['source_commit'], 'code_sha256': json_hash(source_files),
                'source_files': source_files, 'source_archive_sha256': sha256_file(out / 'executed_code.zip'),
                'active_model_promoted': False, 'fresh_test_available': False,
                'interpretation': 'No supported accuracy improvement; the same final configuration and inspected learned parameters were recovered.'}
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    metric_rows = '\n'.join(f"| {r.metric.replace('_', ' ').title()} | {r.V2:.2%} | {r.V3:.2%} | {r.V3_ci_low:.2%}–{r.V3_ci_high:.2%} | No |" for r in table.itertuples())
    ranked = pd.read_csv(after.path / 'final_development_ranking.csv').sort_values('mean_balanced_accuracy', ascending=False)
    new_best = ranked.loc[ranked.candidate_id.str.startswith('v3_')].iloc[0]
    note = f'''# V3 robustness study — 2 October 2026

The expanded study produced **{after.summary['metrics']['balanced_accuracy']:.2%} nested balanced accuracy**, compared with **{before.summary['metrics']['balanced_accuracy']:.2%}** for V2. The difference is **{value:+.2f} percentage points**, with a paired approximate 95% interval of **{lo:+.2f} to {hi:+.2f} points**. This does not establish a reliable improvement. The strict five-metric >75% target remains unmet.

![Measured comparison](../results/research_v3/comparison.png)

## What was tested

All 35 original V2 candidates remained eligible. Twenty fixed additions tested normal quantile transformations, mean/median/standard-deviation feature subsets (222 columns), covariance shrinkage, regularized histogram gradient boosting, reduced-space nearest neighbors and smaller mutual-information feature sets. The development-only tail audit found 11 features with an extreme value more than 20 interquartile ranges from the median. That motivated the experiment without establishing a cause of poor prediction.

`configs/v3_robust.json` was committed before execution. The study retained the original 597 development songs, 52 normalized artist groups, five outer / three inner folds, seeds and native thresholds. All learned transforms were fitted inside training folds. The 55 candidates completed **{len(trials)} recorded fits**, with **{status.get('failed', 0)} failed trials** and **{warning_count} trial warnings**. Runtime was **{execution['seconds']:.2f} seconds** including startup. Two CPU workers and unchanged dependencies were used.

## Results and their limits

| Metric | V2 nested | V3 nested | V3 approximate 95% interval | V3 >75%? |
| --- | ---: | ---: | ---: | --- |
{metric_rows}

The paired procedures changed {comparison['changed_oof_predictions']} predictions and produced only {comparison['additional_correct_oof_predictions']} additional correct predictions overall. Class-1 recall increased while class-0 recall decreased. Whole-artist bootstrap intervals condition on fixed predictions and omit some training and selection uncertainty. The development data had already been inspected in V2, so V3 is an exploratory follow-up, even though its inner/outer folds remain correctly separated.

**The final model configuration stayed `lr_mi_50`**: logistic regression, C=0.1, with 50 selected features. Its final learned imputation, variance, scaling, mutual-information scores, coefficients and intercept matched V2 exactly in the recorded comparisons, and all 597 development predictions and scores were identical. Serialized pipeline bytes have different hashes; numerical equivalence is based on the explicit state and prediction checks, not byte identity. The small nested change concerns the expanded selection procedure across outer folds and does not establish a better final deployed model.

The best newly added full-development tuning setting was `{new_best.candidate_id}` at **{new_best.mean_balanced_accuracy:.2%}** mean fold balanced accuracy, below the incumbent's **{after.summary['tuning_balanced_accuracy']:.2%}**. Tuning scores are not independent performance estimates. No historical labels were evaluated again. No fresh collection exists. The active demo remains `v1_baseline`, with clearly labelled V2 and V3 research options.

## Software, evidence and sources

The new tests check that quantiles do not learn from validation inputs, all 20 additions clone and reload with consistent predictions, the full raw input schema is retained, and internal random early stopping is disabled. A dedicated app check verifies the V3 run identity, 55-setting method description and exploratory qualifier. Exact final test results are in `results/research_v3/final_test_command.json` and `final_tests.log`.

Evidence: `results/v2/v3_nested_001/` contains the immutable run, all trial records and fold memberships. `results/research_v3/paired_comparison.json`, `metrics_comparison.csv`, `preflight.json`, `evidence.json` and the exact executed source archive make the comparison traceable. Original V1/V2 measurements remain preserved. This study is uploaded on branch [`research/chorus-v3`](https://github.com/DhanushSS/chorus-hit-predictor/tree/research/chorus-v3). GitHub authentication was restored on October 3; the update has not been merged into `main`.

Methods: [scikit-learn quantile transformations](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html), [histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html), [discriminant analysis and shrinkage](https://scikit-learn.org/stable/modules/lda_qda.html). These methods can help on some datasets; their documentation does not promise better hit prediction.

## What would enable the next useful experiment

The user confirmed there are no recordings available. The pinned original repository contains no audio files. The inspected ISMIR/MSD release supplies features and different labels, so it cannot substitute for matching chorus recordings. See `V3_DATA_SOURCE_REVIEW.md` for links, access limitations and the recorded source inventory.

The next input is a collection of permitted recordings with recording/version identities, credited performer evidence and verified labels. Re-extract one consistent feature set, compare richer frozen audio representations against a matched baseline, and reserve new data before choosing the final model. Until then, another broad search over these same features has no demonstrated route to 75%. No score or dataset size is promised.
'''
    (ROOT / 'docs/V3_RESEARCH_RESULTS.md').write_text(note)
    print(json.dumps({'trial_statuses': status, 'trial_count': len(trials), 'warnings': warning_count, 'paired': delta}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
