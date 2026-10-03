# Codex implementation brief: Chorus Hit Predictor audit and 75%+ research plan

**Repository:** `https://github.com/DhanushSS/chorus-hit-predictor`  
**Prepared:** October 1, 2026  
**Audited branch:** `main`  
**Audited commit:** `0e9e4628d0a72af79694bc555d64a562e83dcfa2`  
**Audited tree:** `25db599399d402490b220a0e1140f5e348193edf`  
**Audience:** Codex working in a checked-out copy of this repository.  
**Status:** Source-code and committed-artifact review, followed by an implementation specification. No improved model has been trained as part of preparing this document.

This consolidates and supersedes the earlier 70%-target handoff. Read the entire file, reconcile it with the current checkout, and implement the unblocked work. Do not respond with another generic list of machine-learning tips.

---

## 1. Mission and definition of success

Act as a machine-learning engineer, audio-processing engineer, and careful maintainer. Improve the actual code, evidence, and working demo. Aim for **one final model with accuracy, balanced accuracy, positive-class precision, recall, and F1 all strictly greater than 0.75 on appropriate unseen evaluation data**. Class 1 is the year-end-hit class unless an explicitly separate task is approved.

This is an empirical research target, not an instruction to manufacture passing scores. Report every candidate honestly; do not require every algorithm or the dummy baseline to exceed 75%. Preserve balanced accuracy as the primary model-selection metric. Report ordinary accuracy separately. If the user later requires every real classifier to clear the target, evaluate that requirement model by model rather than forcing the table to pass.

Use unrounded values for target checks. A displayed 75% is not necessarily strictly greater than 75%. A point estimate above 75% is also not proof that the population performance exceeds 75%; report uncertainty separately.

### Work autonomously within these boundaries

- Start with the supplied CSV and CPU-capable methods. Do not block available work on missing audio, a paid API, or a GPU.
- Preserve the working demo, original artifacts, author details, attribution, and unrelated local changes. Use a reviewable branch or patch; do not push to `main`, merge, publish data, or spend money without authorization.
- Do not weaken artist separation, relabel mistakes, select favorable seeds, omit failed trials, inflate training scores into test scores, or move the target to an easier task without disclosure.
- A genuinely blocked optional stage must not stop earlier stages. Record the exact blocker and continue useful work.
- Verify actual tool and execution access. Never claim a test, training run, commit, or improvement occurred without evidence.
- Maintain `docs/experiment_log.md` and a machine-readable run manifest so a later Codex session can resume without repeating failed work.

The implementation, search budgets, and acceptance criteria below are proposed engineering work. They are not claims that these changes already improve the repository.

## 2. What the repository audit actually established

### Scope and execution limits

The GitHub connector resolved the commit above and returned its recursive tree with `truncated: false`. Core data/training code, inference code, app code, test source, dependency files, artifact metadata, report/notebook generators, and the upstream label-collection notebook were inspected. The earlier handoff was also read to consolidate the requirements. Sources are indexed in Section 12.

A local `git clone` attempt failed because the execution runtime could not resolve `github.com`. The connector reads succeeded. Therefore:

- The baseline values below were read from committed artifacts, not freshly recomputed from the full CSV.
- The existing pytest suite, saved joblib model, and full training pipeline were **not executed in this audit**.
- The compiled report, presentation, and executed notebook were inventoried; their generators were reviewed, but their rendered outputs were not independently checked here.
- No complete label re-audit, recording verification, or exhaustive discovery of every defect is claimed.

Codex must complete the runtime verification in its own checkout. Do not erase these distinctions from the final report.

### Is everything needed already in GitHub?

| Asset | Audited status | Consequence |
|---|---|---|
| `data/chorus_features.csv` | Tracked, 7,296,094 bytes | Existing-feature experiments can start without original audio. |
| `data/provenance.json` | Tracked | Contains source, hashes, and label/extraction limitations. |
| `models/selected_model.joblib` | Tracked, 1,401,556 bytes | Existing fitted classifier is available; compatibility must be verified before use. |
| Training, prediction, audio, and app modules | Tracked | There is an implementation to extend, not a blank project. |
| Metrics, CV trials, split manifest, test predictions, figures | Tracked | Historical evidence and comparison artifacts are available. |
| `tests/test_project.py` | Six test functions present in source | Existing regression coverage can be run and expanded. |
| Original songs or extracted chorus recordings | No audio files found in the audited tracked tree; provenance explicitly says original audio is unavailable | Learned audio embeddings and verified waveform parity require additional recordings. |
| Pretrained audio-embedding weights or cached embeddings | Not found in the audited tracked tree | No embedding-backed improvement is already implemented. |
| Track-level chart evidence and release-date/observation-window manifest | Not present as a tracked structured audit dataset | Independent label verification and temporal evaluation need enrichment. |
| `.github/workflows` and `AGENTS.md` | Not present in this tree | Automated CI and persistent agent guidance are possible additions. |

This inventory covers the audited commit, not untracked files on the user's computer, other branches, or external storage. The repository contains enough for **CSV-based training and the current demo**, but not all inputs for a new audio-backed study. [R0, R1, R3, R7, R8]

## 3. Baseline to preserve

The recorded dataset contains 751 songs, 518 raw audio features, and 71 normalized artist-name groups. Labels are 385 class-0 songs and 366 class-1 songs. Training uses 597 songs from 52 groups; the old test uses 154 songs from 19 different groups, with 78 class-0 and 76 class-1 songs. [R1]

| Candidate | Training CV balanced accuracy | Test accuracy | Test balanced accuracy |
|---|---:|---:|---:|
| Majority baseline | 50.00% | 50.65% | 50.00% |
| Logistic regression | 52.29% | 52.60% | 52.55% |
| Linear discriminant analysis | 52.81% | 52.60% | 52.60% |
| Linear support vector machine | 49.57% | 55.19% | 55.16% |
| Radial-basis-function support vector machine | 52.83% | 50.65% | 50.52% |
| **Polynomial support vector machine: selected** | **53.63%** | **46.75%** | **46.63%** |
| Random forest | 53.32% | 52.60% | 52.45% |
| Gradient boosting | 51.82% | 53.25% | 53.00% |
| Neural network | 49.87% | 52.60% | 52.68% |

The selected model's precision is 0.4516, recall 0.3684, F1 0.4058, and area under the receiver operating characteristic curve (ROC-AUC) 0.4556. Its approximate 95% artist-bootstrap balanced-accuracy interval is 38.71%-55.80%. Its confusion matrix is:

```text
                   Predicted 0  Predicted 1
Actual 0                44           34
Actual 1                48           28
```

The original selection uses training CV before evaluating test results. Do not retroactively replace its selected model with the linear SVM simply because the latter has the highest observed test accuracy. [R1, R2]

### Strengths that already exist

Preserve fold-local imputation, variance filtering, scaling, optional principal component analysis (PCA), artist-grouped evaluation, metadata exclusion, the majority baseline, split records, and artist-level bootstrap intervals. Tree models already skip PCA. The classes are nearly balanced. These are not new fixes introduced by Codex. [R1, R2, R4]

## 4. Findings and required corrections

### F01. The target is not generic hit versus non-hit

**Evidence:** Class 1 is the upstream year-end Hot 100 collection; class 0 is another sampled weekly Hot 100 song. The pinned upstream `CollectData.ipynb` confirms year-end collection followed by weekly-chart collection. Its `getData` function filters duplicates by title alone and matches artists by their literal names. [R3, R14]

**Implication:** Both classes contain charting songs. Title-only deduplication and literal artist matching create collection risks, but this review does not establish which individual labels are wrong. Unknown or unverified is not synonymous with incorrect.

**Action:** Preserve this target for the first track. Add label definition, evidence status, chart coverage, recording identity, market, and observation-window fields to an audit manifest. Verify a reproducibly selected label sample, then expand coverage as sources permit. Correct only documented errors using a consistent rule. Record unresolved cases without inventing answers.

**Acceptance:** Any changed negative class, success threshold, population, or label rule receives a new task/dataset version and its own baseline. Do not call a change to charted-versus-uncharted an improvement on the original task.

### F02. Dataset diversity is limited; the exact performance bottleneck is unproven

**Evidence:** 597 training songs, 52 training artist-name groups, 518 raw features. Several pipelines reduce dimensions with PCA. Saved candidate CV scores are close to the dummy baseline. [R1, R2]

**Action:** Measure training-versus-validation gaps, retained dimensions, per-group performance, and group-aware learning curves. Diagnose weak label signal, overfitting, distribution differences, or several causes rather than declaring the feature/song ratio to be proof of overfitting. More songs from existing artists are not the same experiment as more independent artist coverage.

**Acceptance:** Measured diagnostics guide the next acquisition or modeling step. The previously suggested 58%-62% range and 2,000-5,000-song target are not validated forecasts or minimum sample-size requirements.

### F03. Representation and classifier searches are limited

**Evidence:** Current non-tree pipelines generally use fixed 95%-variance PCA. The candidate grids are small and do not compare supervised feature selection or Extra Trees. [R2]

**Action:** Compare original features, PCA alternatives, and supervised feature selection on identical grouped folds. Test bounded extensions to regularized classifiers. Retain the original candidates for comparison; do not assume a neural network or boosting library is inherently better.

**Acceptance:** A reproducible ablation table identifies which change helped, if any. Record post-transform dimensions, fit failures, and warnings. Keep every learned transform inside training folds. [D1, D5, D6]

### F04. The old test set has already influenced discussion

**Evidence:** Its results are published in the repository and have been inspected during this project discussion. This does not retroactively invalidate the original selection order. It constrains future claims. [R1, R2]

**Action:** Freeze old membership as a `historical_test` benchmark. Make new modeling decisions within the original training partition, unless an explicitly versioned audit changes the population. Use nested grouped CV for internal assessment. Obtain new, locked evaluation data for fresh final confirmation.

**Acceptance:** Distinguish tuning scores, nested development estimates, historical comparisons, and fresh tests. A new random split of already examined records is not a newly untouched dataset. Do not rotate seeds until a favorable result appears. [D2, D4]

### F05. Artist-string normalization is not complete identity resolution

**Evidence:** `normalize_artist()` performs Unicode normalization, case-folding, and whitespace cleanup. It does not resolve aliases or collaborating performers. Duplicate artist/title checks normalize the artist but not the title. [R4]

**Action:** Audit aliases, featured artists, title variants, duplicate recordings, remasters, and near-duplicates. Use evidence-backed canonical identifiers. For strict unseen-performer claims, keep connected credited-artist and duplicate-recording groups together; inspect whether large connected groups make the design infeasible.

**Acceptance:** Overlap checks match the declared identity level. Do not blindly split names on `&` or `and`, which can be part of real artist names. Preserve versioned mappings and unresolved identities. Changing the grouping definition requires a disclosed new evaluation protocol.

### F06. The audit can hash the wrong dataset

**Confirmed code-level defect:** `load_data(path=...)` accepts a custom path, but `data_audit(frame)` hashes the global `DATA_PATH` regardless of where `frame` came from. An alternate dataset can therefore receive the default file's provenance hash. This does not show that the currently committed default-dataset hash is wrong. [R4]

**Action:** Carry the source path and byte hash explicitly in a dataset context, or change the audit API to take the actual source. Separate source-byte identity from a processed-frame fingerprint. Do not attach a file hash to a modified DataFrame without also recording the transformation/version.

**Acceptance:** A regression test loads two different files and verifies that each audit identifies its own bytes. Missing, mismatched, or modified inputs fail clearly.

### F07. Import provenance and identifiers need safeguards

**Evidence:** `scripts/prepare_data.py --source` accepts a local CSV and validates shape/feature order, but writes the pinned upstream commit and label provenance without checking whether its bytes match that pinned source. It generates `CH0001`, etc., by row position. [R9]

**Action:** For exact legacy reproduction, verify the expected source SHA-256 before writing. For an alternate source, require explicit provenance and a new dataset version rather than claiming the original commit. Preserve the old IDs for the immutable legacy dataset; generate stable, version-aware recording IDs for new collections. Replace critical schema assertions with explicit exceptions where appropriate.

**Acceptance:** A changed source with the same shape is not silently attributed to the pinned source. Row reordering in a new collection does not reassign song identity or corrupt stored splits.

### F08. Upload extraction is not verified equivalent to training extraction

**Evidence:** Training features are upstream precomputations associated with `pychorus`. Uploads use the local repetition heuristic or manual segment selection. The upstream audio and exact extraction environment are unavailable. The extractor deliberately preserves the `kew` spelling and an unusual zero-crossing frame length for compatibility. [R3, R5]

**Action:** Preserve the legacy extractor and its experimental status. Add versioned extraction configuration. Use one shared implementation for all new training audio and uploads. Record segment positions, sample rate, padding, normalization policy, transform parameters, and library versions.

**Acceptance:** New extraction changes require a matching re-extracted training set and retrained model. Do not change only the upload-side zero-crossing calculation. Matching column names or a successful synthetic-tone test does not prove waveform-feature parity. This mismatch is an upload limitation, not by itself an explanation for the old CSV holdout score.

### F09. Model, data, metrics, and caches can become inconsistent

**Evidence:** The app loads separate model, metrics, dataset, and prediction files. App and CLI use global feature columns and labels. They do not enforce a shared run ID, extractor contract, or dataset/model compatibility check before predicting. `assets()` has no run argument for cache invalidation. The bundle already stores some provenance, but that is not consistently enforced at load time. [R2, R6, R7]

**Action:** Introduce a validated artifact loader shared by app, CLI, and report generation. Use an ordered raw-input schema from the bundle; feature selection remains inside the fitted pipeline. Store label definitions, input requirements, run and dataset IDs, extraction version, prediction threshold, and model hash. Key cached assets by immutable run identity.

**Acceptance:** Incompatible models, metrics, or schemas cannot be silently mixed. Upload inputs must match the model's schema/extractor, not the training dataset's file hash. For the bundled historical-song demo, validate the expected dataset/version and evaluation membership too. Show clear compatibility errors.

### F10. Reports and notebook explanations contain stale-on-retrain text

**Evidence:** `scripts/build_notebook.py` embeds the polynomial-SVM conclusion, 46.6%, and the old interval as literal text. `scripts/build_reports.py` hard-codes some counts, the date, parts of the method, a selected-polynomial-SVM heading, and interpretation of the old result. `app.py` also hard-codes 518-feature wording. Some numeric tables are dynamic, but the surrounding prose is not fully dynamic. [R6, R10, R11]

**Action:** Generate model names, feature counts, split details, intervals, and conclusions from a validated run summary. Interpret uncertainty from the actual run instead of always claiming its interval includes 50%. Keep historical baseline text explicitly historical.

**Acceptance:** Swap in a test run with a different model name and schema; every generated label and conclusion stays consistent. Do not update only a table while leaving the old narrative. Rebuild the presentation through an actual supported process; its regeneration is not implemented by `build_reports.py`.

### F11. Training writes into shared baseline locations

**Evidence:** The current trainer writes model, metrics, predictions, split files, and plots into fixed paths. Plotting can leave a prior PCA plot behind when a later selected model lacks PCA. [R2, R12]

**Action:** Use immutable run directories and atomic writes. Preserve the baseline first. Represent a promoted run with an explicit pointer/configuration rather than overwriting historical evidence. Generate plots only for the current run and list them in its manifest.

**Acceptance:** Failed or interrupted training cannot leave a mixture of old and new assets. Reruns do not destroy historical evidence or present stale plots as current.

### F12. Tests protect the baseline but do not cover the expanded system

**Evidence:** Six existing tests check provenance/shape, artist separation, saved-model metrics, synthetic audio/schema, invalid audio/segment bounds, and a Streamlit demo path. Some tests hard-code `(751, 522)` and a `scale` pipeline step. No workflow file is present in the audited tree. No test was run during this review. [R0, R8]

**Action:** Preserve immutable V1 fixture tests, add configurable V2 tests, and separate fast unit tests from slower artifact/audio/app integration tests. Test negative paths for hash mismatches, wrong schema/order, missing run assets, altered labels, new feature counts, unsupported scores, and invalid group support. Add CI when repository workflow permissions and dependencies allow it.

**Acceptance:** Test expectations enforce integrity, not a fabricated 75% threshold. Synthetic fixtures are for software tests only, never evidence of real-song performance.

### F13. Candidate export is not the same as evidence of predictive usefulness

**Evidence:** The trainer excludes the dummy model from the eligible winner list, then exports the highest-CV non-dummy candidate. It reports scores as uncalibrated. [R2]

**Action:** Keep a candidate artifact for research, but add an explicit development-only promotion decision and evidence status. Allow `candidate_not_validated` or `no_supported_improvement`. Do not turn an SVM margin into a claimed commercial-success probability. Evaluate calibration only under appropriate group-aware development splits if probabilities are needed.

**Acceptance:** A functioning demo is distinguished from a validated predictor. Threshold tuning, probability calibration, and ensemble weights never use final-test labels. Report precision/recall tradeoffs rather than promising that a threshold change improves everything. [D3]

### F14. Audio upgrades and forecasting require data not contained in the CSV

**Evidence:** The tracked assets do not provide original recordings, embedding caches, or the track-level time/evidence manifest needed for a forecasting evaluation. [R0, R3]

**Action:** Build a versioned ingestion contract and optional frozen-audio-embedding path, then run them only with real, permitted recordings. Define release dates, prediction cutoff, market, and outcome window for any forecasting variant. Treat audio-plus-metadata as a separately named product with different inputs.

**Acceptance:** No embeddings are invented from the 518 summary columns. No later stream count, chart position, or popularity score is used to reveal the target. Retrospective classification is not presented as proven future-hit prediction.

## 5. Implementation sequence

### Phase A - Preserve, reproduce, and complete the audit

1. Read current repository instructions, run `git status`, and record the current commit. Compare it with the audited commit; newer evidence wins.
2. Preserve the dataset, model, metrics, splits, predictions, and their hashes in a baseline snapshot before running write-producing scripts. Do not reset user changes.
3. Create an isolated environment, inspect pinned dependencies, and attempt installation. Verify availability and compatibility; do not silently rewrite version pins or load the model under incompatible versions and claim reproduction. Saved pickle/joblib artifacts require trusted provenance and appropriate software compatibility. [D7]
4. Run the existing tests and record exit codes, logs, duration, and environment. Reproduce V1 in a disposable worktree or copy so original outputs survive.
5. Independently recompute metrics from saved predictions; then compare with loaded-model predictions and retraining results. Check row alignment, positive-class mapping, score orientation, hashes, split membership, duplicates, and feature order.
6. Inventory the actual full CSV and any local audio visible in the authorized project directory. Do not search unrelated personal storage. Compute label/group distributions and validate the recorded dataset audit rather than copying it.
7. Implement F06/F07 and artifact-safety fixes with regression tests before making alternate-data experiments.

**Outputs:** `docs/accuracy_audit.md`, preserved baseline manifest, test/reproduction logs, and a status for every finding: confirmed, resolved, not reproduced, or still unverified.

**Continue immediately into Phase B when its inputs are available.** Missing original songs do not prevent this phase.

### Phase B - Controlled experiments on the existing feature table

Keep original task, dataset version, and historical-test membership fixed for the comparable track. Fix any verified data error through a separately versioned dataset and rerun its own baseline.

Implement a budgeted, configuration-driven runner rather than a giant unbounded grid. Suggested initial modes, subject to hardware:

| Mode | Purpose | Suggested budget, not a performance promise |
|---|---|---|
| Smoke | Exercise interfaces and artifacts | Tiny fixture; never reported as real accuracy |
| Quick | Explore the existing training partition | Three grouped folds; at most about 12 configurations per candidate family |
| Thorough | Estimate the declared selection procedure | Five outer grouped folds, three inner grouped folds, bounded predeclared search |

Make budgets configurable with trial, wall-time, and parallelism limits. A timeout must preserve completed results and mark the run incomplete. Do not turn skipped configurations into successful runs. Use consistent fold assignments for paired comparisons.

#### Candidate representations

- Original V1 preprocessing as a control.
- Imputation, constant-feature filtering, suitable scaling, and no reduction.
- PCA alternatives such as 30/50/100 components or 90%/95% retained variance, validating fold-specific limits.
- `SelectKBest(f_classif)` and seeded mutual-information selection with a bounded subset of `k = 30, 50, 100, 150, 200, "all"`.
- Optional predeclared family ablations, such as Mel-frequency cepstral coefficients (MFCCs) plus energy/spectral features, rather than an uncontrolled search over every subset.

Supervised selection can help or hurt. Seed stochastic scoring and serialize named callables, not fragile local lambdas. Clamp or reject invalid feature-count requests after filtering and log that decision. Do not fit a selector once on the entire dataset. [D5, D6]

#### Candidate classifiers

Retain the original dummy, logistic regression, linear discriminant analysis, SVM variants, random forest, gradient boosting, and neural-network controls. Add Extra Trees and modest extensions to regularization/feature selection. Histogram gradient boosting, XGBoost, or another dependency is optional after a justified comparison, not required for the first runnable version.

For a bounded search, sample sensible regularization scales, tree depth/leaf sizes, and RBF width parameters. Store the exact spaces and sampling seeds before evaluation. Scaling is useful for some estimators, but do not force PCA onto trees. Compare class weights rather than assuming severe imbalance exists.

Keep early stopping, calibration, feature learning, and stacking validation group-aware. An estimator's default internal random validation split must not quietly violate the declared artist protocol. Disable unsupported internal splitting or supply a documented grouped alternative.

#### Nested evaluation requirements

In each outer development fold, all tuning decisions, including candidate family/representation selection, must occur within that outer fold's training groups. Fit its chosen pipeline there and produce predictions only for its outer validation groups. Outer validation labels must not tune thresholds, ensemble weights, or feature subsets.

Store both per-family results and the outer predictions from the **selection procedure**. Do not choose a family from outer scores and present that best outer mean as an unbiased estimate of the newly selected family. Any subsequent redesign using those results is another development iteration. Nested CV helps assess a procedure but does not make unlimited adaptive experimentation independent. [D2]

For final development selection, use the declared rule on the full development partition after the procedure is fixed. Use inner-fold balanced accuracy as the primary selection score. Keep threshold selection separate and properly nested when enabled. Record training scores and all validation predictions for diagnosis.

**Outputs:** Immutable run manifests, fold assignments, trial tables, out-of-fold predictions, learning curves, representation ablations, candidate artifacts, and a measured comparison. Distinguish a tuning CV maximum from a nested estimate.

### Phase C - Better data and audio representations

This stage has additional input requirements. Implement its interfaces and tests even when full training is blocked; do not claim actual embedding results without audio.

#### Data contract

For new records, capture at least:

```text
recording_id, dataset_version, title, canonical_artist_ids,
original_artist_credit, release_date, recording_version,
label, label_definition_version, label_evidence_status,
label_source, chart_market, observation_start, observation_end,
audio_path_or_authorized_reference, audio_sha256, audio_rights_basis,
extractor_version, segment_start_seconds, segment_duration_seconds,
artist_group_id, duplicate_group_id
```

Store unresolved fields explicitly. Do not invent release dates or classify recently released songs as failures before their observation window is complete. Audit acquisition failures by class and artist; a missing-audio subset can change the population. Matching tracks using names alone requires ambiguity checks.

The upstream code/feature-table license is not evidence that the user has rights to redistribute each recording. Record the actual permitted source and keep licensed audio private when required. Do not bypass access controls or automatically download large copyrighted collections.

Collect in documented batches with greater artist coverage, then inspect learning curves. No fixed number of songs guarantees 75%. Keep a new final-evaluation collection segregated from development and lock its inclusion rules before model decisions.

#### Shared extraction and multi-segment experiments

Implement a single versioned extraction path for new dataset preparation and upload inference. Preserve the legacy path. Consider manual chorus, deterministic repeated-segment selection, and a separately declared multi-segment variant. Use the same selection policy across classes and partitions.

Keep every segment, augmentation, duplicate, and linked artist group in one partition. Aggregate at song level for scoring. Do not count three excerpts of one song as three independent test songs. Log segment positions, preprocessing, and audio hashes. [D4]

#### Frozen pretrained music embeddings

One candidate is `m-a-p/MERT-v1-95M`. Its official model card describes audio feature extraction, 24 kHz pretraining, custom model code, and a CC BY-NC 4.0 license. Verify the exact revision, processor configuration, intended-use compatibility, and runtime needs at implementation time. It is not a ready-made hit predictor and is not evidence of 75% accuracy on this dataset. [D8]

Start with frozen embeddings plus a small regularized classifier. Load the appropriate processor, honor its sampling rate, document hidden-state/layer pooling, and cache embeddings by audio hash, segment definition, model revision, and extraction configuration. Do not guess the processor settings from the old 22,050 Hz extractor. Review and pin any custom code before enabling remote-code execution.

Compare handcrafted-only, embedding-only, and combined features on the same eligible recordings and splits. Fit any data-dependent scaling, reduction, layer combination, or fine-tuning within training folds. Frozen, per-record feature extraction can be cached; learned corpus-level transforms cannot see held-out data. Record possible pretraining exposure to evaluation recordings when unknown, and qualify claims accordingly.

Fine-tuning and ensembles are later experiments, not prerequisites. Keep CPU fallback and optional dependency isolation. State download size and compute needs before large jobs.

**Outputs:** Validated ingestion manifest, shared extractor, optional embedding cache, labeled-audio coverage report, and controlled ablations. Record exactly which missing recordings, evidence, rights, or compute block an unrun experiment.

### Phase D - Thresholds, target checks, and final evidence

Decide the threshold-selection objective before evaluating. The default experiment uses ordinary estimator thresholds. An optional constrained objective can investigate precision and recall above 0.75 on group-aware development predictions while maintaining useful balanced accuracy; infeasible constraints must return an explicit failure, not a manufactured threshold.

All threshold/calibration choices belong inside the development evaluation procedure. A changed threshold trades off errors; it cannot create information absent from the features. Do not calibrate on final-test data or interpret arbitrary SVM scores as probabilities. [D3]

Freeze the final model, input schema, group definition, and threshold before evaluating the fresh set. Evaluate the old benchmark separately and label it historical. Keep evaluation models separate from any later production refit that uses all labels.

Report accuracy, balanced accuracy, class-1 precision/recall/F1, class-0 recall, macro-F1, ROC-AUC when defined, confusion matrix, sample counts, group counts, and per-group diagnostics. Include uncertainty and the target status of each metric. Handle one-class folds and undefined metrics explicitly; do not silently delete unfavorable folds.

For paired historical comparisons, compare predictions on the same records and resample whole groups consistently. Report fold variation as fold variation, not automatically as a confidence interval. Repeated folds are not independent new test datasets. Document bootstrap assumptions and instability from few groups.

Write target status with an evaluation qualifier, for example:

```json
{
  "threshold": 0.75,
  "comparison": "strictly_greater",
  "evaluation_status": "nested_development",
  "target_met_in_this_evaluation": false,
  "fresh_test_available": false,
  "fresh_test_target_met": null
}
```

The values above illustrate a schema, not a computed result. Set actual fields from actual evaluations. Do not conflate a development target pass with fresh-test confirmation.

### Phase E - Demo, reports, and reproducibility

Keep the current target wording for the original task. Display run identity, model name, schema, score semantics, target status, and evaluation status. Show upload limitations until the corresponding end-to-end audio pipeline has been evaluated.

Generate reports and notebook conclusions from the common validated summary. Make the page-count/layout checks survive additional models through explicit layout handling, not smaller unreadable text. Rebuild and inspect exported documents and the presentation where the necessary tooling exists; list any stale exports rather than claiming they were updated.

Provide one reliable setup path, quick-run path, thorough-run path, and app-start path. Update short agent instructions only after checking existing `AGENTS.md` files. A short pointer to this brief plus the experiment log is preferable to silently replacing existing project instructions. Codex supports project guidance through `AGENTS.md`; do not assume an arbitrary Markdown filename is automatically loaded. [D9]

## 6. Suggested file responsibilities

These are proposed additions or refactors, not all currently existing interfaces. Prefer small reusable modules over duplicating the old training code.

| Path | Responsibility |
|---|---|
| `chorus_hit/audit.py` | Dataset context, source/hash verification, identity and manifest checks |
| `chorus_hit/data.py` | Robust loading and legacy-compatible validation |
| `chorus_hit/evaluation.py` | Frozen group folds, nested selection, metric calculation, target status |
| `chorus_hit/train_v2.py` | Budgeted experiment orchestration and resumable run logging |
| `chorus_hit/artifacts.py` | Validated run bundle loading, compatibility, and promotion metadata |
| `chorus_hit/audio.py` | Legacy extraction plus shared versioned extraction entry points |
| `chorus_hit/embeddings.py` | Optional frozen-embedding extraction and cache |
| `chorus_hit/predict.py`, `app.py` | Same validated artifact and prediction interface |
| `scripts/prepare_data.py` | Exact legacy-source checks and honest alternate-source provenance |
| `scripts/build_reports.py`, `scripts/build_notebook.py` | Run-derived text, metrics, and interpretations |
| `results/v2/<run_id>/` | Immutable configuration, folds, trials, predictions, metrics, figures, logs |
| `models/v2/<run_id>/` | Model bundle and its checksums, separate from V1 |
| `docs/accuracy_audit.md`, `docs/experiment_log.md` | Evidence, decisions, unresolved issues, and restart instructions |
| `tests/`, optional `.github/workflows/tests.yml` | Unit, integration, baseline, and CI coverage |

An artifact manifest should capture code commit, dirty-tree state, dataset/source hashes, label/group/extractor versions, raw input schema, training identifiers, fit population, model revision, package versions, score/threshold semantics, trial budget, evaluation status, metric definitions, and generated outputs. Include all these identities in caches where relevant.

## 7. Command-line interface to implement

The existing repository supports the legacy training, prediction, and app commands. The V2 commands below are a **requested interface**, not commands claimed to exist now. Implement them, test `--help`, and report the exact supported syntax if names differ.

```bash
# Proposed: audit a specific file and record its actual provenance.
python -m chorus_hit.audit --data data/chorus_features.csv --out results/audit

# Proposed: bounded existing-data exploration; do not inspect historical test.
python -m chorus_hit.train_v2 --config configs/v2_quick.json --run-id v2_quick_001

# Proposed: predeclared nested grouped evaluation.
python -m chorus_hit.train_v2 --config configs/v2_thorough.json --run-id v2_nested_001

# Existing test command; expand tests without silently deleting V1 coverage.
python -m pytest -q

# Existing app command; adapt run selection through a documented configuration.
python -m streamlit run app.py
```

Configuration should control dataset/version, fold manifests, identity grouping, candidate search space, predeclared seeds, max trials, timeout, parallelism, artifact paths, optional embeddings, and explicit historical/fresh evaluation mode. Require deliberate final-evaluation invocation rather than scoring the holdout on every trial.

Do not run legacy training or report-generation commands in the only copy of the original artifacts before preservation.

## 8. Acceptance tests and evidence requirements

### Software and data integrity

- [ ] Original dataset and artifacts are preserved and hashed.
- [ ] Custom-file audits identify the actual source; processed data have separate versioned fingerprints.
- [ ] Alternate imports cannot inherit false pinned provenance; source identity is checked before writes.
- [ ] Track IDs, feature/label alignment, class mapping, and split membership are stable and verified.
- [ ] Group and duplicate overlap tests run at every relevant evaluation layer.
- [ ] Learned preprocessing sees only fitting rows; valid feature/PCA sizes are enforced.
- [ ] Model reload reproduces predictions; missing, extra, reordered, or incompatible inputs have a declared policy.
- [ ] App, CLI, metrics, and reports identify the same run; a run change invalidates stale caches.
- [ ] Interrupted runs do not overwrite the baseline or create mixed assets.
- [ ] Report/notebook model names, counts, and conclusions change correctly with the run.
- [ ] Audio tests cover silence, non-finite samples, short/overlong input, segment bounds, sample-rate conversion, and decoder failures.
- [ ] Optional embedding tests verify cache identity, schema, sampling-rate handling, and missing-audio behavior.

### Evaluation integrity

- [ ] Every reported number has a dataset, task version, group definition, partition, and evaluation status.
- [ ] Metrics can be recomputed from saved per-song predictions and documented score orientation.
- [ ] Search spaces, seeds, budgets, failures, warnings, and incomplete runs remain visible.
- [ ] Neither historical nor fresh test labels tune the model, thresholds, calibration, or ensemble weights.
- [ ] The selection procedure is evaluated separately from the best tuning score.
- [ ] Fresh-test status is unknown/null until a genuine fresh evaluation occurs.
- [ ] Per-class support, group support, and uncertainty accompany the headline accuracy.
- [ ] A changed task/population or recording subset has a new baseline, not a misleading before/after claim.
- [ ] Synthetic data never contribute to real performance claims.
- [ ] Failure to exceed 75% is reported honestly without changing the evaluation to force success.

Software tests must not require a particular empirical score. A separate result checker may accurately report target pass/fail, but the implementation must remain valid when performance is below target.

## 9. How to decide the next experiment

Use development evidence, not wishful forecasts:

| Observation | Next investigation |
|---|---|
| Training strong, grouped validation weak | Regularization, smaller representation, artist dependence, additional independent groups |
| Training and validation both weak | Label validity, representation quality, task signal, or underfitting |
| CSV scores improve but uploads remain unreliable | Recording/extractor parity and end-to-end audio evaluation |
| One artist or one split drives the gain | Group support, alias leakage, uncertainty, and sensitivity under declared repeats |
| New embeddings help on the same recordings | Confirm with controlled ablations and an untouched evaluation set |
| New data change labels or the population | Version the task/data and rerun a matched baseline |
| Existing-data improvements remain small | Continue ingestion/representation work; do not endlessly search seeds |

These observations motivate investigations; none alone establishes a causal explanation. Keep failed but informative experiments in the log.

## 10. Required final Codex handoff

Return:

1. A status table separating **implemented and tested**, **implemented but not run**, **blocked**, and **not attempted**.
2. Confirmed findings, fixes, and remaining hypotheses, with repository paths and run evidence.
3. Changed files and a reviewable patch/branch; no claim of a pushed commit unless it was actually pushed with authorization.
4. Exact commands executed, dependencies, test results, timings, and error logs.
5. Before/after comparisons using matched tasks and evaluation definitions. Separate tuning CV, nested development estimates, historical tests, and fresh tests.
6. Artifact locations and the run selected for the demo, or an explicit decision not to promote a candidate.
7. Target status for accuracy, balanced accuracy, precision, recall, and F1, including uncertainty and whether fresh confirmation exists.
8. The concrete missing inputs for any blocked audio/label stage, not a vague request for "more data."
9. A beginner-friendly explanation for the user's project viva: what changed, why it was tried, what the measured result means, and what it does not establish.

End each work session by updating the experiment log with the next unblocked action. Do not stop after planning when implementation and execution are available. Conversely, do not label scaffolding as a completed experiment.

## 11. Important corrections to earlier discussion

- 75% is the current target, not a guaranteed result. Neither the reference paper nor this repository proves that every model can exceed it.
- The earlier numerical expectations about attainable accuracy and sufficient dataset size were not measured. Do not repeat them as forecasts.
- The reported 46.75% is ordinary test accuracy; 46.63% is test balanced accuracy; 53.63% is training-CV balanced accuracy. They are not interchangeable.
- The 518 raw columns are not the final input dimension of every current model; PCA is already used in several pipelines.
- The low score has no single established cause. Confirmed engineering defects and representation/data hypotheses must stay distinct.
- The repository already has meaningful leakage safeguards. Preserve them rather than claiming to introduce them for the first time.
- A different negative class or an audio-plus-metadata product changes the research question or interface. It is not a silent replacement for the existing audio-only classifier.
- Absence of source recordings blocks new embedding measurements, not auditing, existing-feature experiments, artifact improvements, or test expansion.

## 12. Evidence and primary references

### Immutable repository evidence

Append each relative path below to this immutable base URL:

```text
https://github.com/DhanushSS/chorus-hit-predictor/blob/0e9e4628d0a72af79694bc555d64a562e83dcfa2/
```

| Ref | Path or resource | Evidence |
|---|---|---|
| R0 | Recursive Git tree `25db599399d402490b220a0e1140f5e348193edf` | Tracked inventory; response says `truncated: false` |
| R1 | `results/metrics.json`, `results/model_comparison.csv` | Recorded baseline, counts, selected model, confidence intervals |
| R2 | `chorus_hit/train.py` | Pipelines, search, model selection, metrics, bootstrap, artifact writes |
| R3 | `data/provenance.json` | Dataset source, hashes, label definition, unavailable recordings/environment |
| R4 | `chorus_hit/data.py` | Normalization, data validation, groups, default-path hashing defect |
| R5 | `chorus_hit/audio.py` | Segment selection, legacy feature calculations, audio validation |
| R6 | `app.py` | Separate asset loading, cache behavior, hard-coded feature wording, demo claims |
| R7 | `chorus_hit/predict.py` | Global schema/label usage and CLI inference behavior |
| R8 | `tests/test_project.py` | Six existing tests and legacy-specific assumptions |
| R9 | `scripts/prepare_data.py` | Local-source import, provenance assignment, row-position identifiers |
| R10 | `scripts/build_reports.py` | Dynamic tables mixed with hard-coded narrative; PDF-only generation |
| R11 | `scripts/build_notebook.py` | Notebook creation/execution and hard-coded old conclusion |
| R12 | `chorus_hit/plots.py` | Fixed figure locations and conditional PCA output |
| R13 | `requirements.txt`, `requirements-dev.txt`, `README.md` | Pinned dependencies and documented workflow/scope |
| R14 | Upstream `CollectData.ipynb` at commit below | Actual year-end/weekly collection and title-only duplicate filtering |

Upstream immutable source:

```text
https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus/blob/838e76f96f7aa755a882b4d2581afe1e0ad3f8a9/CollectData.ipynb
```

Legacy hashes recorded by the repository; independently verify them in the runtime audit:

```text
Upstream CSV SHA-256:
b1e12d084dc503bf5e71c28c406bbc33e30bcc1f9000250b293d1b2b32851f0a

Prepared CSV SHA-256:
79951ffc393fc836405509326d1ee03efd96434327d800711c4b8ed817ef44ce
```

### Official methodological and model documentation

Consulted October 1, 2026. Verify exact version/API compatibility before implementation. These sources explain methods and interfaces, not an expected hit-prediction score.

- **D1 - scikit-learn, Common pitfalls:** `https://scikit-learn.org/stable/common_pitfalls.html`
- **D2 - scikit-learn, Nested versus non-nested CV:** `https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html`
- **D3 - scikit-learn, Decision threshold tuning:** `https://scikit-learn.org/stable/modules/classification_threshold.html`
- **D4 - scikit-learn, Cross-validation and grouped data:** `https://scikit-learn.org/stable/modules/cross_validation.html`
- **D5 - scikit-learn, Feature selection:** `https://scikit-learn.org/stable/modules/feature_selection.html`
- **D6 - scikit-learn, Mutual information classifier scores:** `https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html`
- **D7 - scikit-learn, Model persistence:** `https://scikit-learn.org/stable/model_persistence.html`
- **D8 - MERT-v1-95M official model card:** `https://huggingface.co/m-a-p/MERT-v1-95M`
- **D9 - OpenAI, Project instructions with AGENTS.md:** `https://developers.openai.com/codex/guides/agents-md`

---

**Start now with preservation and runtime verification, then implement the existing-data improvements. Pursue the 75%+ target through measured experiments, not by changing the meaning of the score. Deliver working code and evidence, while keeping blocked upgrades and unverified results explicit.**
