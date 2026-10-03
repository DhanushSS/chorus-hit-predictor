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

## 2026-10-03 — Hugging Face feature pilot declared

- User requested measured results using Hugging Face. Hub repository inspection and public Dataset Viewer calls succeeded; the plugin dataset-search endpoint had earlier reported disabled by server configuration.
- Pinned Music4All metadata from `Leon299/music4all` revision `a391160e3e17f351d5ab2d05439a7d3d7f0440eb`. Found 434 unique normalized artist/title candidate links across the complete existing table; these are NOT verified recording identities.
- Located the authors' companion Music4All-Onion release, Zenodo record 15394646 (CC BY 4.0), containing a 50-dimensional musicnn feature table. Downloaded this file and verified publisher MD5 `080a17808e1e86c849f3a4d99ad72b64`. No audio, MERT weights, paid compute or private data uploaded.
- Locked `configs/hf_feature_pilot.json` before fitting: three fixed logistic-regression arms (legacy MI50, musicnn, combined), C=0.1, native threshold, V3 outer-fold membership restricted to candidate-linked development rows, paired whole-artist bootstrap. No tuning, historical evaluation or promotion.
- This is a linkage feasibility pilot. Different recording versions, segment coverage, external pretraining overlap and inherited label uncertainty prevent treating its numbers as verified repeated-chorus accuracy or a fresh confirmation. Artist/title metadata is used only for joining, never as model features. Spotify popularity and acoustic metadata are excluded.
- **Next unblocked action:** run the declared pilot and report all three arms on the same rows; retain missing-input and source limitations. Current user preference is private study, so this new work will remain local.

## 2026-10-03 — Hugging Face linked-feature pilot completed locally

- Run `hf_features_001` completed all 15 fixed fits in 5.40 seconds, with no warnings/failures. 340 candidate-linked development songs, 46 artist groups, 201 positive / 139 negative labels; no cross-group exact feature duplicates detected. Historical rows were excluded.
- Matched legacy baseline: accuracy 54.41%, balanced accuracy 50.79%. Published musicnn: accuracy 56.47%, balanced accuracy 52.09%. Combined: accuracy 57.35%, balanced accuracy 54.17% (approximate interval 49.00%–59.22%).
- Combined-minus-legacy paired balanced-accuracy gain +3.38 percentage points, approximate interval -0.22 to +7.21; inconclusive. All arms have ordinary accuracy below the 59.12% always-positive reference on this class mix. Musicnn's 76.12% positive recall alone does not meet the five-metric target.
- Independently recomputed metrics, verified all 20 immutable run assets and five source hashes, reloaded all 15 models and reproduced predictions/scores. Verified artist separation, original-development membership and training-only musicnn scaler means. Results remain exploratory owing to unverified recording versions, clip/chorus equivalence, performer identities, inherited labels and pretraining overlap.
- Saved source inventories, pinned download/hash helper, input subset, fold artifacts, logs and report `docs/HUGGING_FACE_PILOT_RESULTS.md`. Bulk source tables are locally retained and Git-ignored. No MERT inference, new audio extraction, paid compute or publication occurred; V3 and active demo results are unchanged.
- **Next unblocked action:** inspect the saved linkage candidates for recording/version evidence and seek authorised matching audio through the original dataset authors or faculty. New matched-segment/fresh-test work still requires those inputs. Do not present this 340-song pilot as a direct gain over V3's 597-song result.

## 2026-10-03 — Supplied degradation report: reconciliation and stability protocol

- User requested continuation of the supplied report and explicitly kept year-end hit versus other charted song as the main task. No target redefinition is authorised for the primary study.
- Source inspection confirms regularization existed in V1; V2/V3 add nested selection, feature selection and robust ablations. Artist credits lack explicit collaborator fields; no actual cross-fold identity leak is established by the report's hypothetical examples. Evidence-backed strict grouping exists but canonical IDs remain unresolved.
- Locked `configs/stability_audit.json` and `scripts/run_stability_audit.py`: frozen `lr_mi_50` and training-fold majority baseline, original 597 development rows, 10 seeds 0–9, five grouped folds each, 100 fits total. No new tuning, historical scoring, model promotion or favourable-seed selection.
- Repeated measurements reuse songs and are descriptive fold-assignment sensitivity. No independent-repeat confidence interval, significance test, or fresh-evaluation claim will be made. The report's 1,000-shuffle proposal needs its null and full selection procedure defined before execution; it is not part of this fixed-candidate stability run.
- **Next unblocked action:** execute the fixed stability study and reconcile all 14 report items with current source/artifact evidence, keeping unresolved identity/label/audio issues explicit.

## 2026-10-03 — Stability and report reconciliation completed

- `stability_001`: ten fixed grouped partitions of the original 597 development songs / 52 artist groups; 100 completed fits, zero warnings/failures, 24.25 seconds. Frozen LR MI50 mean pooled BA 53.0525%, descriptive SD 1.5533 points, range 49.0279%–54.2587%; majority baseline 50% BA. Candidate beats dummy in 9/10 repeats; 0/10 meet all five strict >75% checks. No significance claim or best-seed selection.
- Checked actual artist strings and normalized-title similarities: 71 groups, zero explicit collaboration-credit strings under the recorded marker scan, zero same-artist title pairs at SequenceMatcher >=0.90, zero fully evidenced canonical performer records. Identity leakage and label noise remain unresolved rather than confirmed or ruled out. No groups/labels changed.
- Reconciled E-01 through E-14 against source/artifacts. Nested evaluation, regularization, feature selection and robust ablations were already implemented. Retained assigned target per explicit user answer; identified unsupported causal claims, arbitrary acceptance thresholds and invalid repeat-independent inference in the pasted report.
- Full suite: 26 passed / 4 existing dependency/decoder warnings in 14.80 seconds. New evidence tests independently recompute BA, check development membership and artist separation, reload all 100 estimators, and verify training-only scaling. Preserved all 26 protected baseline files, V1/V2/V3 runs and HF pilot hashes.
- Saved `docs/ACCURACY_REPORT_RECONCILIATION.md`, immutable run records, original supplied report, review inventory and logs. Current branch `research/evaluation-stability`; no public upload, historical re-evaluation or demo/model promotion.
- **Next unblocked action:** review the corrected audit and resolve recording/performer/chart evidence for an authorised audio subset. The assigned task remains primary. Selection-aware permutation testing and optional additional models remain explicitly unrun; new matched-audio/fresh-test work still requires verified inputs.

## 2026-10-03 — Latest studies uploaded and original-paper comparison

- User explicitly requested uploading the latest local work. Verified the existing repository remains public and fast-forwarded `research/chorus-v3` to include the HF pilot, ten-seed stability study and report reconciliation. Existing PR #1 includes the update; main is not merged.
- Inspected Eric Liu's original PDF, Table 2: preferred NN test accuracy 58%, highest listed test accuracy RF 68%, NN CV accuracy 62%. The different labels, datasets and evaluation protocols prevent a superiority claim. Added `docs/ERIC_LIU_COMPARISON.md`.
- **Next unblocked action:** verify GitHub CI for the updated PR and review the comparison. Matching recording/label evidence remains necessary for further audio experiments.
