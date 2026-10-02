# Experiment log

## 2026-10-01 — Phase A

- Checkout matches audited commit `0e9e4628d0a72af79694bc555d64a562e83dcfa2`; initial tree clean.
- Working branch: `audit/chorus-v2`; no remote writes authorized for this work.
- Original complete Git archive preserved alongside project; every tracked file hashed in `results/baseline/v1_manifest.json`.
- Original environment is an isolated Python 3.12 virtual environment; verifying exact pins before loading trusted local model.
- Next: existing tests, independent metric checks, and disposable V1 reproduction.

### Phase A completed

- Existing tests: 6 passed; new audit/artifact tests: 6 passed.
- Full disposable V1 rerun: 49.76 seconds, all recorded metrics and historical predictions reproduced.
- F06/F07 fixed and tested; baseline wrapped as validated immutable run `v1_baseline`.
- Predeclared both quick and nested configurations (35 candidates, fixed seeds). No historical evaluation during tuning.
- Next: complete runner smoke/checkpoint tests, then execute quick and nested development experiments in that order.

## Phase B completed

- `v2_quick_001`: 35 configurations, 105 tuning fold fits plus 9 learning-curve fits; 16.39 seconds. Selected `lr_mi_50` (logistic regression, C=0.1, 50 mutual-information features). Mean fold tuning BA 0.5287898627; pooled tuning OOF BA 0.5284061552. These are tuning scores.
- `v2_nested_001`: same predeclared configurations; 5 outer / 3 inner folds, 689 recorded fits including per-family outer assessments and diagnostics; 77.18 seconds. Procedure pooled nested BA 0.5285016287, approximate group-bootstrap CI [0.4825816149, 0.5734497717]. No historical scores used. Final full-development choice remains `lr_mi_50`.
- Learning curves for the chosen configuration: mean train/validation BA at about 9 groups 0.815/0.495; 18 groups 0.701/0.514; 35 groups 0.651/0.529. This shows a train/validation gap and sensitivity to group coverage, without proving one cause of low performance.
- Both completed runs are immutable and contain configuration, fold IDs, every trial, warnings/failures, OOF records, family assessments, group diagnostics and model reload evidence. No automatic promotion; keep `v1_baseline` for the demo.
- Next in supplied order: Phase C ingestion contracts, shared extraction and optional embedding interfaces/tests. Real-audio experiments remain blocked by absent recordings and rights/identity evidence.

## Phase C interface work completed; real-data experiments blocked

- All 751 records now have an explicit recording/label/time/rights audit manifest; unknowns remain null. Coverage is 0 eligible recordings, reported by label and artist.
- Strict connected identity grouping requires evidence-backed canonical IDs and duplicate links. Current matched CSV evaluation retains artist-string grouping; 20 near-duplicate candidates are flags only, with no labels/groups changed.
- Six deterministic sample records: three positive labels corroborated by MusicBrainz series/release records. Three negative labels remain unresolved because weekly participation does not establish year-end exclusion. Official year-end pages returned HTTP 402. No labels changed and no source recording identities verified.
- Shared extraction supports preserved `legacy-librosa-518-v1` and separately named `shared-librosa-518-v2`; only V2 uses a 2048-sample zero-crossing frame, requiring matched re-extraction/retraining. Demo stays on legacy. Configurations include sampling/padding/normalization/statistics/library versions.
- Optional MERT local adapter and content-addressed cache implemented. Synthetic fixture tests validate cache keys, model sampling rate and missing-audio behavior. The real model is NOT loaded. Model revision pinned from official commit page to `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`; official processor uses 24 kHz and normalization. Official PyTorch weight size 378 MB; no weights downloaded. Local API metadata lookup timed out (retained log); web commit lookup succeeded afterward.
- Three audio/ingestion/cache tests passed. Real ingestion, MERT feature generation, embedding ablations and a fresh evaluation remain blocked by concrete missing inputs listed in `docs/AUDIO_DATA_REQUIREMENTS.md`.

## Phase D completed for available data

- Native estimator thresholds fixed before experiments. No threshold optimization or calibration attempted.
- Explicit historical evaluation of frozen `v2_nested_001`: 154 rows / 19 groups, balanced accuracy 0.5516194332; CI [0.4713308828, 0.6150822829]. V1 matched benchmark was 0.4662618084.
- Paired historical difference +0.0853576248; whole-group 95% interval [-0.0158763635, 0.1685604993] includes zero. This does not establish a reliable improvement.
- Accuracy 0.5519480520, precision 0.5479452055, recall 0.5263157895, F1 0.5369127517. All five strict >0.75 checks fail in both nested and historical evaluations. Fresh-test result remains null.
- Demo promotion decision remains unchanged: keep `v1_baseline`. Next: Phase E validated app/CLI/report integration, expanded tests and reviewed exports.

## 2026-10-02 — Phase E and final handoff

- App, CLI and reports use the shared validated run loader. Original model remains active; V2 is a visibly labelled research option. Browser verification confirmed a historical prediction and matching V2 nested results. Percentage-point changes are labelled correctly.
- Final full suite: 20 passed / 4 dependency-decoder deprecation warnings; 51.39 seconds in pytest, 54.38 seconds wall. Fast suite: 12 passed / 8 deselected. Prediction and help commands all exit 0. Relevant app test rerun after final display adjustment. Exact commands/logs: `results/audit/final_commands.json` and `app_final_check_command.json`.
- All 803 trial records across the two real CSV studies completed; zero trial warnings/failures. Controlled fixture timeout tests are software tests, not failed empirical trials. New CLI process-group supervision was added after the measured runs; a forced full-study timeout has not been executed. Remote CI is configured and unrun.
- Updated two-page and one-page PDFs, executed notebook (six code cells, no errors) and 10-slide editable presentation saved under V2 paths. All PDF pages and slide renders reviewed. Final deck imported/rerendered with matching pixels and portable validation passed. No native PowerPoint opening claimed. Initial export precision error and its repair are retained.
- Baseline data, models, scores, exports, notebook and attribution remain byte-identical; complete baseline archive verified. Exact source producing the V2 experiments preserved separately. See `preservation_check.json`, `v2_executed_code_identity.json`, `export_manifest.json` and `session_manifest.json`.
- Full handoff: `docs/AUDIT_HANDOFF.md`; local review branch `audit/chorus-v2`. Package and patch are prepared for review; no push/merge/publication. Neither nested nor historical evaluation meets any of the five strict >75% checks. Fresh status remains unavailable/null. No promotion.
- **Next unblocked action:** review the package and rehearse the supplied demo using `docs/VIVA_V2.md`. Complete the permitted recording, identity and label evidence in `docs/AUDIO_DATA_REQUIREMENTS.md` before a new matched audio study; the missing inputs block real embedding and fresh-test work.

## GitHub update authorized — 2026-10-02

- The user requested uploading the completed update to their existing GitHub repository. Verified `origin` as `DhanushSS/chorus-hit-predictor`, with write access and `main` still at the preserved baseline.
- Prepared branch `audit/chorus-v2` for upload and a review pull request. Main remains unchanged; model promotion remains unchanged. Original measured run files and initial handoff evidence are preserved.
- Updated current navigation to distinguish the initial local handoff from the subsequent GitHub update.
- **Next unblocked action:** verify the uploaded branch and pull request, inspect GitHub Actions, and resolve any concrete CI failure before reporting completion.

### GitHub publication verified

- Uploaded branch `audit/chorus-v2`; remote commit `2dfe0ee29aaa77bf0b24ec0701cc4ec6b01165e6` matched the local checkout. `main` remained at `0e9e4628d0a72af79694bc555d64a562e83dcfa2`.
- GitHub Actions run `36970412334` completed successfully for that publication commit: https://github.com/DhanushSS/chorus-hit-predictor/actions/runs/36970412334 . Local tests and empirical results are unchanged.
- No pull request was created: GitHub CLI returned HTTP 401 (invalid stored login); the connector returned HTTP 403 (integration cannot create pull requests); the browser requires sign-in. No credentials or permission settings were changed.
- **Next unblocked action:** review the uploaded branch and use the existing viva guide. Opening a pull request requires a restored GitHub login with pull-request write permission, or manual creation through GitHub. Main is not merged and no model is promoted.

## 2026-10-02 — V3 robustness study declared before execution

- User authorized further accuracy work and confirmed they have no recordings. Inspected upstream repository and a public ISMIR/MSD dataset; neither supplied matched source recordings under the current task. No external data were silently substituted.
- Development-only diagnostic: 11 raw features have an observed extreme more than 20 interquartile ranges from the median. This motivates robust transformations without proving that outliers caused low performance.
- Locked `configs/v3_robust.json`: all 35 V2 controls plus 20 fixed new settings (quantile transforms, core statistics, shrinkage LDA, regularized histogram boosting, small MI subsets and reduced-space neighbors). Same seeds, outer/inner folds, labels and native thresholds; 55 candidates, 2 workers, 1,800-second budget.
- Reusing previously inspected development folds means this follow-up is exploratory. There is no independent confirmation. No historical re-evaluation is planned for this study.
- **Next unblocked action:** verify transformation isolation and serialization, then execute this one declared nested study and compare paired development predictions.

## 2026-10-03 — V3 completed and verified

- Executed the declared 55-setting nested study as `v3_nested_001`, using source commit `542413979fad33217f228a35d4801c61ac5348aa`. All 1,059 trial records completed, with zero trial warnings/failures; 129.87 seconds wall time.
- Nested balanced accuracy 0.5325227451 (V2 0.5285016287). Paired change +0.0040211165, approximate artist-bootstrap 95% interval [-0.0270901427, 0.0304199863]. This does not establish a reliable gain; all five >75% checks remain false and fresh-test status remains unavailable.
- Final selection again chose `lr_mi_50`. Inspected imputation, variance, scaling, MI scores, coefficients and intercept exactly match V2, as do all 597 development predictions and decision scores. Serialized model hashes differ; numerical equivalence is based on the recorded state checks. No historical evaluation was repeated.
- Source audit found no audio paths in the complete pinned upstream tree. The reviewed ISMIR/MSD release has features and different labels. No audio or substitute dataset was acquired. User confirmed no recordings are available.
- Full suite: 24 passed, four dependency/decoder deprecation warnings; 13.60 seconds pytest / 13.94 seconds wall. Browser verified V3 identity, 53.3% display, uncertainty interval, exploratory qualifier and unmet target. Screenshot and structured checks are saved under `results/research_v3/`.
- Preservation verified: 26 protected baseline files, the complete baseline archive, 857 previously tracked V2 run/report/notebook files and all four completed run manifests. Original active demo remains `v1_baseline`.
- Added reproducible comparison script, exact executed-source archive, figures, paired evidence and written findings. Local branch `research/chorus-v3`; no V3 publication claimed. Earlier GitHub authentication failure remains unresolved.
- **Next unblocked action:** review the V3 findings and demo. For a new audio study, assemble permitted recordings with verified recording identities, performer links and labels, then reserve new evaluation data before model selection. Restored GitHub authentication is needed to upload this local update.

## 2026-10-03 — GitHub access restored and V3 uploaded

- User restored GitHub authentication and requested another check. Verified the active account as DhanushSS and successfully pushed `research/chorus-v3` to the existing `DhanushSS/chorus-hit-predictor` repository.
- Initial uploaded V3 commit: `36d403d5ae2d5840931fc9ed9dc7e09f274dbcee`. The update includes the V2 audit, V3 source/configuration, completed trials, model artifacts, demo and measured findings. Main remains at the preserved baseline.
- Updated publication notes to reflect the successful upload. Prior package receipts describe their creation-time state; experimental evidence is unchanged.
- **Next unblocked action:** open the review pull request and verify GitHub Actions for the uploaded update. A new accuracy study still needs the recording and label inputs described in the V3 findings.
