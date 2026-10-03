# Accuracy report reconciliation: chorus-hit-predictor

**Date:** 3 October 2026

**Task:** Year-end hit versus other charted song, retained as the main project at the user's explicit request.

**Basis:** Current source, completed V1/V2/V3 and Hugging Face pilot artifacts, and a newly executed stability study. The supplied report was README-based and did not inspect source. Its full text is preserved in `results/report_reconciliation_001/supplied_report.txt`.

## 1. Summary

The supplied report identifies useful risks, but mixes historical and current model results and presents several hypotheses as established causes. The recorded numbers do **not establish accuracy degradation over time**: V1 historical testing, V3 nested development evaluation, and the smaller Hugging Face pilot evaluate different procedures or populations.

The new ten-seed check of the frozen `lr_mi_50` candidate produced **53.05% mean pooled out-of-fold balanced accuracy**, descriptive between-seed SD **1.55 percentage points**, and range **49.03%–54.26%**. It beat the training-fold majority baseline's 50% balanced accuracy in **9/10 repeats**. No repeat met all five >75% checks. Reusing previously inspected data and a previously chosen model makes this a stability diagnostic, not independent confirmation.

A 9/10 comparison is not a binomial significance test: repeats share songs and training data. The seed range is not a confidence interval, and no favourable seed was selected. The existing V3 score and demo remain unchanged.

| Evidence | Population/procedure | Balanced accuracy |
|---|---|---:|
| Original V1 | Historical 154-song test; polynomial SVM | 46.63% |
| V2 | Nested selection, original 597 development songs | 52.85% |
| V3 | Expanded nested selection, same 597 songs | 53.25% |
| HF pilot, combined inputs | 340 candidate-linked development songs | 54.17% |
| New stability diagnostic | Frozen LR MI50, same 597 songs, 10 grouped partitions | 53.05% mean |

The HF pilot's matched legacy baseline is 50.79%; its gain remains inconclusive and recording/segment equivalence is unverified. It is not a direct comparison with V3. A weak signal, limited artist coverage, model variance and label uncertainty are plausible contributors; their individual causal effects and the maximum achievable accuracy are unknown.

## 2. All 14 report items checked

| ID | Finding after source/evidence review | Status and action |
|---|---|---|
| E-01 | Both classes charted; the assigned label definition is documented. Its intrinsic difficulty is not quantified. | Keep the assigned target. Do not substitute never-charted songs or regression. |
| E-02 | Upstream labels remain incompletely verified. Prior six-song review corroborated three positives via secondary sources; three negatives unresolved. | Retain evidence status and labels. New same-artist normalized-title similarity review found zero pairs at 0.90; this limited check cannot verify chart labels. |
| E-03 | Raw inputs are 518-dimensional, but final LR already selects 50 features inside folds. V3 tested 222 core statistics and smaller subsets. | Relevant dimensionality interventions are already tested. Feature/sample ratio alone is not proof of overfitting; retain measured learning curves. |
| E-04 | The original historical test has only 19 artist-name groups. | V2/V3 nested grouped assessment already implemented. A new test remains missing. Repeated CV cannot create independent artists or guarantee a three-point interval. |
| E-05 | Names are normalized strings, with incomplete credited-performer identities. | Checked 71 names: no explicit collaboration strings under the recorded marker scan; zero records with evidenced canonical artist IDs. Cross-fold performer leakage remains unresolved, not demonstrated. Do not split names blindly. |
| E-06 | Summarized features discard much temporal detail. | HF musicnn/combined pilot is measured but has unverified recording links and different segment coverage. Proper MERT and beat/dynamics comparisons need matching permitted recordings. |
| E-07 | Tail audit motivated robust transforms; V3 tested quantile transforms and core statistics. | Already tested without reliable gain. Standardization changes scales before PCA, so domination by raw variance cannot simply be assumed. Fold-local correlation pruning remains an untested option, not an established remedy. |
| E-08 | No explicit full-track repetition feature vector is trained. RMS and spectral statistics already provide some energy/timbre information. | Full-track repetition, chorus position and LUFS work needs source audio and annotations. No original duration/tempo/recording evidence was invented. |
| E-09 | Upload/training extraction parity remains unverified. | Both legacy and separately named 2048-frame shared extraction versions exist. Preserve legacy compatibility; do not silently change upload extraction without matching training features. |
| E-10 | Reliable recording release dates, completed outcome windows and a fresh chronological collection are missing. | No valid temporal evaluation yet. Chart date is not interchangeable with recording release date. |
| E-11 | V1 selection was inside development CV; subsequent nested selection assesses wider searches. | Calling the V1 winner 'selected by chance' is not established. New fixed-candidate stability includes a training-fold majority baseline. A full selection-aware permutation study remains unrun. |
| E-12 | V1 already tunes C, shrinkage, tree settings and neural-network weight decay. V2/V3 add MI/ANOVA/PCA and bounded candidates. | 'No effective regularization/selection' is outdated. Elastic net is an untested candidate; random internal early stopping would violate grouped validation unless redesigned. |
| E-13 | Earlier principal nested study used one declared outer seed. | Executed ten fixed seeds for the frozen incumbent: all scores retained, no new model ranking/search. This addresses fold sensitivity, not repeat-independent inference. |
| E-14 | External non-audio information could change predictive signal and interpretation. | No established audio-only ceiling. Metadata/lyrics experiments need suitable sources, outcome cutoffs, rights and declared scope; popularity measurements can leak outcomes. |

## 3. New stability experiment

Protocol: `configs/stability_audit.json`; runner: `scripts/run_stability_audit.py`. Seeds **0–9** were fixed before execution. Each uses five artist-grouped folds on the original **597 development rows / 52 groups**. The candidate is frozen logistic regression C=0.1 with 50 fold-selected MI features, estimator seed 42. Imputation, variance filtering, scaling and feature selection fit only the training portion. There is no new tuning or candidate selection. The baseline learns the training-fold majority.

**100 fits completed**: 50 candidate fits and 50 dummy fits. All saved estimators reproduce their recorded validation predictions after reload. Historical data and active demo model are untouched. Each song appears once per repeat, so 5,970 prediction records represent **597 unique songs**, not 5,970 independent samples.

| Measure across ten pooled out-of-fold repeats | Candidate | Majority baseline |
|---|---:|---:|
| Mean accuracy | 53.18% | 51.42% |
| Mean balanced accuracy | 53.05% | 50.00% |
| Balanced-accuracy range | 49.03%–54.26% | 50.00%–50.00% |
| Between-seed BA standard deviation | 1.55 points | 0.00 points |
| Mean precision | 52.00% | 0.00%* |
| Mean recall | 48.48% | 0.00% |
| Mean F1 | 50.14% | 0.00% |

*The majority baseline predicts class 0 in all folds; its positive precision has no predicted positives and is conventionally recorded as zero. The unweighted fold-mean BA averaged across repeats is 53.16%; it differs from pooled BA because folds have different sizes and class compositions. Both are saved explicitly.

The report's proposed ≤1-point between-repeat SD condition was not met. That threshold is a requested diagnostic target, not a correctness requirement or an established prerequisite for valid ML. No seed or record was discarded to make it pass. Model-ranking stability among multiple tuned families was not tested by this frozen-candidate experiment.

## 4. Corrections to proposed verification rules

- Do not strip 'remix', 'version', featured performers or punctuation and then merge/relabel automatically. These can identify distinct recordings or real artist names. Similarity flags need evidence-backed adjudication.
- A rising learning curve suggests potential value from additional data; it does not prove a single bottleneck. A fixed 3,000-song / 300-artist target cannot guarantee accuracy or interval width.
- Repeated folds overlap. Ordinary independent-repeat confidence intervals, an unqualified Wilcoxon test over repeats, and a '9 out of 10' significance claim are not justified here.
- A within-artist label permutation preserves each artist's label mix and tests a conditional null. Two development artists have only one label, which cannot change under that shuffle. The remaining 50 are mixed-label. A permutation test of the selected pipeline must also account for the selection process; shuffling saved predictions is insufficient. See [scikit-learn's documented permutation test](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html).
- StandardScaler followed by PCA cannot be diagnosed from raw marginal variances alone. Small post-transform skew, fixed train-validation gaps, and nonzero permutation importance are not guarantees of better unseen accuracy.
- A score over 75% warrants checking data, splits and provenance; it is not automatically proof of leakage. Conversely, a score below 75% is not proof the software is incorrect.
- The existing acceptance checks require all five metrics strictly greater than 75%. The supplied report's single-metric ≥75% wording is different; neither is reached by the present evidence.

## 5. What remains and why

**Unresolved data evidence:** full credited-performer IDs and duplicate links, recording-level label verification, permitted matching audio, and new outcome-complete evaluation songs. The old label sample is not a fully adjudicated random 100-song audit, so a contamination-rate Wilson interval would be misleading; none is reported. Zero corrections means no confirmed corrections were made, not that label noise is zero.

**Unrun optional studies:** selection-aware 1,000-permutation null, repeated nested ranking of all families, elastic net/correlation pruning, lyric/genre/year ablations and temporal holdout. These are not marked completed. The new result supports prioritizing data/identity verification over another wide search on already examined rows.

**Next step:** obtain recording and chart evidence for a versioned audit subset, resolve credited artists into connected groups, then compare matched-segment audio representations. The assigned target stays primary. No automatic relabeling, target substitution, public upload or demo promotion occurred.

## 6. Evidence and reproduction

- `results/stability_001/`: 100 fold models, 50 fold definitions, all predictions, per-repeat metrics, summary and hashed manifest.
- `results/report_reconciliation_001/`: identity/title checks, artist inventory, original supplied report, execution and verification logs.
- `tests/test_stability_evidence.py`: independent metric checks, development-only/disjoint-fold checks, model reload predictions and training-only scaler means.
- Existing evidence: `results/audit/label_sample_evidence.json`, `results/v2/`, `results/research_v3/`, `results/hf_features_001/` and associated reports.

To reproduce in a disposable copy with the completed stability run absent, use the existing pinned Python 3.12 environment and run `python scripts/run_stability_audit.py`. The runner refuses to overwrite an existing run. The main project test suite verifies the saved evidence.
