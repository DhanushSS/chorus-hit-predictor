"""Export aggregate HSP-S results and figures, never raw features or private paths."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chorus_hit.hit_nonhit.common import read, sha
from chorus_hit.hsp_s.data import verify


def report(split, run, destination, ci_url):
    split, run, out = Path(split), Path(run), Path(destination)
    sm = verify(split, 'split'); mm = verify(run, 'model')
    summary, audit = read(run / 'summary.json'), read(split / 'audit.json')
    detail, diagnostics = read(split / 'split.json'), read(run / 'diagnostics.json')
    locked = read(run / 'locked_evaluation.json') if (run / 'locked_evaluation.json').exists() else None
    if locked and locked['model_manifest_sha256'] != sha(run / 'manifest.json'): raise ValueError('Evaluation/model mismatch')
    out.mkdir(parents=True, exist_ok=False)
    safe = {'task_id': summary['task_id'], 'status': locked['status'] if locked else 'TRAINED_DEV_ONLY',
            'source_doi': '10.5281/zenodo.5383858', 'source_license': 'CC BY 4.0', 'source_sha256': sm['source_sha256'],
            'source_md5': sm['source_md5'], 'source_rows': audit['source_rows'], 'accepted_rows': audit['accepted'],
            'class_counts': audit['class_counts'], 'exclusions_by_reason': dict(Counter(r['reason'] for r in audit['exclusions'])),
            'groups': audit['groups'], 'largest_group': audit['largest_group'], 'feature_count': summary['feature_count'],
            'split': {k: {a: b for a, b in v.items() if a != 'ids'} for k, v in detail.items()},
            'selected_candidate': summary['selected_candidate'], 'training': summary['resubstitution_training'],
            'nested_development': summary['nested_development'], 'nested_group_interval': summary['nested_group_interval'],
            'candidate_comparison': summary['candidate_comparison'], 'diagnostics': diagnostics,
            'locked_test': {k: v for k, v in locked.items() if k not in {'uuid', 'label', 'predicted', 'scores'}} if locked else None,
            'raw_audio_downloaded': 0, 'chorus_features_extracted': 0, 'test_access': 'USED_ONCE' if locked else 'NOT_USED',
            'config_sha256': sha(run / 'config.json'), 'model_sha256': sha(run / 'model.joblib'),
            'split_manifest_sha256': sha(split / 'manifest.json'), 'ci_url': ci_url,
            'model_storage': 'Local ignored results/hsp_s/run_001; public repository contains reproducible code and aggregate evidence',
            'research_scope': 'Publisher-labelled retrospective whole-recording feature benchmark; not a repeated-chorus or future-success model'}
    trials = [read(p) for p in (run / 'trials').glob('*.json')]
    safe['trials'] = {'completed': sum(r['status'] == 'complete' for r in trials), 'failed': sum(r['status'] != 'complete' for r in trials),
                      'seconds': sum(r['seconds'] for r in trials),
                      'with_warnings': sum(bool(r.get('warnings')) for r in trials),
                      'warnings': dict(Counter(w for r in trials for w in r.get('warnings', []))),
                      'failure_reasons': dict(Counter(r.get('error', '') for r in trials if r['status'] != 'complete'))}
    (out / 'results.json').write_text(json.dumps(safe, indent=2, sort_keys=True) + '\n')
    comparisons = sorted(summary['candidate_comparison'], key=lambda x: x['balanced_accuracy'] if x['balanced_accuracy'] is not None else -1)
    fig, ax = plt.subplots(figsize=(10, 10))
    names = [c['candidate_id'] for c in comparisons]
    values = [100*c['balanced_accuracy'] if c['balanced_accuracy'] is not None else np.nan for c in comparisons]
    ax.barh(names, values, color=['#168568' if n == summary['selected_candidate']['id'] else '#376797' for n in names])
    ax.axvline(50, color='#666666', linestyle='--', linewidth=1)
    ax.set(xlabel='Mean inner-fold balanced accuracy (%)', title='HSP-S · development-only model selection', xlim=(0, 100))
    ax.tick_params(axis='y', labelsize=8); fig.tight_layout(); fig.savefig(out / 'model_comparison.png', dpi=150); plt.close(fig)
    if locked:
        matrix = np.array(locked['metrics']['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(matrix, cmap='Blues')
        for (i, j), v in np.ndenumerate(matrix): ax.text(j, i, str(v), ha='center', va='center', fontsize=18, color='white' if v > matrix.max()*.55 else '#172635')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['No chart entry', 'Chart entry'], yticklabels=['No chart entry', 'Chart entry'], xlabel='Predicted', ylabel='Publisher label', title='HSP-S · one-use held-out test')
        fig.tight_layout(); fig.savefig(out / 'confusion_matrix.png', dpi=160); plt.close(fig)
    def percent(value): return f'{value:.2%}'
    metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'macro_f1', 'roc_auc', 'average_precision']
    rows = []
    for key in metrics:
        values = [safe['training'].get(key), safe['nested_development'].get(key), locked['metrics'].get(key) if locked else None]
        rows.append('| ' + key.replace('_', ' ') + ' | ' + ' | '.join(percent(v) if v is not None else 'Not available' for v in values) + ' |')
    body = f'''# HSP-S results and completion report

**Status: {safe['status']}**. Separate experiment approved on 10 October 2026.

## Dataset and method

- Source: **7,736 published HSP-S rows**; whole-recording Essentia features.
- Accepted: **{audit['accepted']}** — {audit['class_counts'].get('1',0)} charted / {audit['class_counts'].get('0',0)} without chart entries.
- Excluded: {audit['source_rows']-audit['accepted']} duplicate/conflicting records; reasons are in `results.json`.
- {audit['groups']} connected artist/recording/album groups; largest component {audit['largest_group']} songs.
- Development: {detail['development']['rows']} songs / {detail['development']['groups']} groups.
- Locked holdout: {detail['locked']['rows']} songs / {detail['locked']['groups']} groups.
- {summary['feature_count']} numeric scalar audio descriptors. Metadata, chart outcomes, listening counts and high-level classifier outputs excluded from predictors.
- 27 preregistered model configurations; five outer / three inner group folds; seed 20261008.
- Missing-value handling, scaling, feature filtering and PCA fitted inside training folds only.
- Downloaded audio: **0**. New 15-second chorus features: **0**. This is not the 278-song chorus experiment.

## Results

Selected configuration: `{summary['selected_candidate']['id']}`.

| Metric | Training (optimistic resubstitution) | Nested development | Locked test |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

The nested result estimates model selection across unseen connected groups. Final
candidate selection uses development folds only. Training metrics are not evidence
of generalization. Fixed native thresholds were used; scores are uncalibrated.

Final-test access: **{safe['test_access']}**. Review flags: `{diagnostics['review_flags']}`.
Group-bootstrap intervals and per-class results are in `results.json`.
See [completion and verification details](handoff.md) for the app check, exact
test commands, remaining blockers and interpretation of the bias diagnostics.

![Development model comparison](model_comparison.png)

''' + ('![Held-out confusion matrix](confusion_matrix.png)\n\n' if locked else '**No held-out confusion matrix: final evaluation was not permitted.**\n\n') + f'''
## Verification and reproduction

Source publisher MD5: `{sm['source_md5']}`. Full source SHA-256:
`{sm['source_sha256']}`. No partial download was trained on.

Trial fits: {safe['trials']['completed']} completed, {safe['trials']['failed']} failed;
{safe['trials']['seconds']:.1f} accumulated trial seconds. Failed fits are retained,
not converted to zero scores or silently removed. The final development model
comparison and warnings are preserved locally; aggregate comparison is published.
{safe['trials']['with_warnings']} trial fits emitted warnings; the warning messages
and counts are retained in `results.json`. Warnings do not mean failed fits.

[Preregistered protocol and exact CLI commands](../hsp_s_protocol.md).
[GitHub verification run]({ci_url}). The report's handoff records the checked SHA
and test results; this link identifies the exact remote workflow rather than an
older baseline check. Model and detailed local artifacts remain in ignored
`results/hsp_s/run_001`; the public repository contains code and safe aggregates.

Streamlit has a separate **Hit vs Non-Hit feature study** page. A genuinely evaluated
compatible model can classify 1–100 records with its exact ordered feature CSV.
It cannot accept MP3s under the original librosa extractor. The original audio-upload
project and its model remain intact.

## Limitations and attribution

The source labels come from publisher matching of Billboard Hot 100 records
(11 August 1958–6 July 2019) to MSD, with sampled non-hits. Missing chart fields
are not independent proof of never charting. The convenience sample, artist/genre
composition, incomplete identity tags and recording-source differences limit
generalization. Group isolation covers known source identities, not every possible
unrecorded alias or cover relationship. Codec/version/duration and permutation
diagnostics are published in `results.json`; review flags prevent test consumption.

This is retrospective classification. It does not predict future commercial
success, establish performance on new audio, or demonstrate superiority to Eric
Liu or the historical model on different datasets and evaluation protocols.

Dataset: Michael Vötter, Maximilian Mayerl, Günther Specht and Eva Zangerle (2021),
*Novel Datasets for Evaluating Song Popularity Prediction Tasks*,
[DOI 10.5281/zenodo.5383858](https://zenodo.org/records/5383858), **CC BY 4.0**.
Changes for this study: scalar-audio feature subset, documented exclusions,
connected-group partitioning, nested model selection and the evaluation above.
'''
    (out / 'README.md').write_text(body)
    return safe['status']


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ['split', 'run', 'out', 'ci-url']: p.add_argument('--' + name, required=True)
    args = p.parse_args(); print(report(args.split, args.run, args.out, args.ci_url))
