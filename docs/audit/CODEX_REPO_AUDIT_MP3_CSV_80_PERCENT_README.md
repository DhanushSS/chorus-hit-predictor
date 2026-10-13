# CODEX MASTER ISSUE AUDIT & EXECUTION PLAN
## Chorus Hit Predictor: MP3 + CSV Inference, Honest 80% Research Goal, Better Streamlit UX, and Professional GitHub README

**Repository:** https://github.com/DhanushSS/chorus-hit-predictor  
**Audited branch:** `main`  
**Audited HEAD:** `c166dddf3f0e639ba39783edf9b67eb959f6e065` (11 Oct 2026)  
**Original archived repository:** https://github.com/DhanushSS/chorus-hit-predictor-archive-2026-10-11  
**Execution owner:** Codex working in the user's actual checkout  
**Deliverables:** tested code, reliable upload flows, honest research experiments, visual UI and README, source-linked issue resolution, final completion report  
**IMPORTANT:** This document is a *read-only repository audit and implementation specification*, not a claim that the requested fixes have already been implemented or that 80% accuracy has been reached.

---

# 0. Non-negotiable first reading

**Read the entire document and inspect the actual current HEAD before changing code.** Some referenced line numbers may shift; paths and code quoted below are from the audit HEAD. If `main` has advanced, reconcile this list with the new code and record changes to the issue inventory. Do not blindly overwrite newer work.

## 0.1 Current verified project state (not a bug list)

| Area | Observed repository evidence as of audited HEAD |
|---|---|
| New project focus | HSP-S retrospective Billboard chart-association benchmark; *whole-recording Essentia features*, NOT the original 15-second chorus project |
| Dataset | 7,736 publisher records; 7,728 accepted after 8 duplicate/conflicting exclusions; 3,865 positives + 3,863 negatives |
| Predictor input | 436 numeric low-level / rhythm / tonal features; excludes chart/popularity fields and artist metadata as model inputs |
| Model | RBF SVM, C=1.0, gamma=scale, class_weight=balanced |
| Development | 6,235 songs and 2,106 connected groups; 5 outer / 3 inner grouped folds |
| Held-out test | 1,493 songs and 527 groups; final locked test used ONCE |
| Test accuracy | **73.07%** (0.7307434695) |
| Test balanced accuracy | **73.17%** (0.7316929339) |
| Hit precision / recall / F1 | **70.42% / 75.35% / 72.80%** |
| Confusion matrix | TN=553, FP=226, FN=176, TP=538 |
| ROC-AUC | 0.8076 |
| Nested development accuracy / balanced accuracy | 70.97% / 70.94% |
| In-sample training accuracy | 79.60% -- optimistic; *not* held-out accuracy and must never be advertised as achieved 80% |
| Existing tests | Cleaned current project reports **26 passed**, and CI for audited HEAD succeeded; old archived pre-cleanup suite was 285 tests |
| Packaged model | `models/hsp_s_run_001.zip` (13,654,495 bytes), with pinned checksums and guarded installer |
| Input supported by app today | Exact ordered HSP-S 436-feature **CSV only**, 1-100 rows and 5 MB max |
| MP3 / WAV prediction | **NOT implemented** for current HSP-S model |
| Original 278-song chorus experiment | Preserved in archive; zero permitted audio and zero new chorus-feature rows |
| Saved model and methodology | Research performed on whole-song descriptors, not MP3 uploads or repeated choruses |

**Citation/evidence within repository:**
- `README.md`
- `app.py`
- `chorus_hit/hsp_s/{data.py,modeling.py,train.py,evaluate.py,download.py}`
- `docs/hit_nonhit/hsp_s_protocol.md`
- `docs/hit_nonhit/hsp_s_results/{README.md,handoff.md,results.json}`
- `docs/repository_migration.md`
- `models/{README.md,bundle.json}`
- `tests/{test_current_project.py,hit_nonhit/test_hsp_s.py}`
- `.github/workflows/tests.yml`
- Last verified GitHub Actions run: https://github.com/DhanushSS/chorus-hit-predictor/actions/runs/38152138607

**Audit classification legend:**
- **CONFIRMED GAP:** directly supported by the current code, UI, or reports.
- **RESEARCH LIMITATION:** explicitly present in results or a scientific reproducibility risk; do not pretend it is a code crash.
- **DESIGN RISK:** plausible failure inferred from code; reproduce before claiming a bug.
- **ENHANCEMENT:** improvement request, not evidence the existing implementation is broken.

The audit was performed from GitHub source, model reports, and CI outcomes; the auditor did **not** personally download the 210 MB training Parquet, retrain the 27 candidates, or perform a real MP3 inference run. Codex must do hands-on verification.

---

# 1. Issue inventory, ordered by priority

## P0 / release-critical: upload functionality and model integrity

### ISSUE-001 -- Current app does not accept MP3 files [CONFIRMED GAP, P0]

**Evidence:** `app.py` uses `st.file_uploader(..., type=['csv'])`, and the app explicitly states MP3/chorus predictions are unsupported. `chorus_hit/hsp_s/evaluate.py::predict_frame` accepts a DataFrame of **exactly 436 HSP-S scalar columns**, not audio samples.

**Fix:** Implement a clearly labeled **Audio Upload** experience for `.mp3` first; optional `.wav` and `.flac` later. A file upload interface alone is NOT a working classifier. Audio may produce a model prediction **only** when a validated, compatible feature extractor supplies all 436 expected features in the correct order and distribution (see ISSUE-002 and Section 2).

**Acceptance:** Valid, permitted MP3 goes through actual decode -> full-recording Essentia-compatible features -> feature audit -> classifier (if proven compatible); otherwise the UI must explicitly say `Audio prediction is not validated/available`, show available audio analysis or a next action, and NEVER fabricate a hit score. Test positive and negative paths. The user should be able to upload an MP3 without breaking the existing working CSV flow.

### ISSUE-002 -- MP3-to-HSP-S feature compatibility is unproven [CONFIRMED GAP, P0]

**Evidence:** HSP-S training vectors are precomputed **whole-recording Essentia/AcousticBrainz** descriptors; `chorus_hit/audio.py` and the archived repeated-chorus extractor produce unrelated **518 librosa** descriptors. No current MP3 -> 436 HSP-S feature mapping exists. No original track audio was downloaded for the current model.

**Fix:** Research reproducible **Essentia MusicExtractor / streaming extractor** invocation, and reconstruct the HSP-S lowlevel/rhythm/tonal 436 scalar feature contract, including Essentia version(s), statistics, parameter profile, mono/stereo handling, resampling, tempo/key algorithms and field units. Match exact field names and types. Keep `.mp3` bytes local and temporary. Extractor must either produce **an honestly comparable 436-feature vector** or reject inference, never zero-fill/rename a 518-feature vector into 436 names.

**Source guidance:** Official Essentia docs: https://essentia.upf.edu/tutorial_extractors_musicextractor.html and https://essentia.upf.edu/streaming_extractor_music.html ; note AcousticBrainz discontinued new submissions in 2022: https://musicbrainz.wordpress.com/2022/02/16/acousticbrainz-making-a-hard-decision-to-end-the-project/ . Do not depend on a hypothetical new-submission API.

**Acceptance:** Publish `docs/audio_upload/feature_compatibility.md` detailing the pinned extractor/profile, **schema coverage 436/436**, parity/drift comparisons on legitimately obtained reference recordings with known source descriptors where possible, confidence limitations, unsupported keys, and the precise decision to enable/disable MP3 predictions. A matching column list alone is NOT parity evidence. If proper parity is unavailable, keep the CSV classifier working and provide a separate non-predictive MP3 preview or pursue a separately evaluated audio model -- never claim existing 73.17% for MP3.

### ISSUE-003 -- Global Streamlit 5 MB upload ceiling is too small for many MP3 files [CONFIRMED GAP, P0 if audio route enabled]

**Evidence:** `.streamlit/config.toml` has `[server] maxUploadSize = 5`; app CSV uses an additional `5 * 1024 * 1024` guard.

**Fix:** When MP3 mode exists, increase server global upload limit thoughtfully (e.g., 25 MB for MP3, not unlimited), while **retaining a separate 5 MB CSV limit**. Enforce duration and decoded-size caps, including sample-rate/channel bounds, format sniffing, timeouts, resource cleanup, and a clear unsupported-file error.

**Acceptance:** Typical under-limit MP3 is accepted; oversize/malformed/too-long files are rejected early; CSV stays capped at 5 MB; no temporary audio remains on disk after success/failure; no uploaded audio is committed, logged in bulk or sent to outside APIs without explicit consent.

### ISSUE-004 -- Artifact verification is coupled to unrelated UI changes [CONFIRMED DESIGN RISK, P0]

**Evidence:** `chorus_hit/hsp_s/train.py::implementation()` hashes **every `chorus_hit/hsp_s/*.py` file**, plus shared training files. `chorus_hit/hsp_s/evaluate.py::load_model()` requires `manifest['implementation'] == implementation()`. Consequently modifying `hsp_s/ui.py` (or adding a new `.py` in that package) can invalidate the existing packaged model even if the SVM itself is unchanged.

**Fix:** Preserve the existing evaluated model's SHA-backed identity and integrity. For new upload/UI modules, prefer **a new module/package outside the globbed paths**, e.g. `chorus_hit/audio_upload/` or `app_components/`, plus `app.py`. If there is a strong reason to update hashed code, use a **versioned, backward-compatible manifest verifier and regression-tested model migration**; never skip integrity checks or silently regenerate a manifest to make a failing model look valid.

**Acceptance:** After every code change, `python scripts/setup_model.py` and saved four-row predictions still pass on a fresh clone; the historical bundle hash, model outputs, 73.17% report and one-use test evidence are unchanged. Include regression test demonstrating that safe frontend-only edits do not brick the model.

### ISSUE-005 -- Exact-order CSV flow is correct but unfriendly [CONFIRMED UX GAP, P1]

**Evidence:** `predict_frame` checks `list(frame.columns) == bundle['columns']` and 1-100 rows, uses `pd.to_numeric` and finite-value guards, then outputs class text and **uncalibrated SVM margins**. Most users cannot create 436 columns manually.

**Fix:** Keep strict validated contracts, but add: two clear upload modes; one-click downloadable template; sample file; preview/shape; duplicate/missing/extra column list; explicit explanation of expected 436 features and correct column order; helpful guidance for old 518-feature legacy CSVs; row-by-row error reporting and safe CSV output. Optionally offer an explicit validated reorder if same unique set of fields, never silently ignore or manufacture columns. Preserve 1-100 rows or add an independently tested batch strategy.

**Acceptance:** Valid example CSV classifies; shuffled/missing/duplicated/non-numeric/Inf/all-empty/oversized/wrong-schema CSV gets a human-readable, actionable error without stack traces. Display score as `decision margin`, **not probability** or `80% confidence`.

### ISSUE-006 -- Training examples in demo may be mistaken for unseen validation [CONFIRMED COMMUNICATION GAP, P1]

**Evidence:** `data/README.md` states `data/example_features.csv` contains **four development (training) rows**; current app calls it a demonstration. This is fine but must never support the generalization score.

**Fix:** Make the app and README call the four-row example a **functionality demo, NOT independent accuracy evidence**. Do not display original true labels as proof of predictive performance unless clearly contextualized.

**Acceptance:** Sample predictions match saved expected outputs; a `Demo data - seen during training` note is visible; research results come from the sealed 1,493-song evaluation.

## P1 / scientific validity and the 80% goal

### ISSUE-007 -- The model has not achieved 80% held-out accuracy [CONFIRMED GAP RELATIVE TO USER GOAL, P1]

**Evidence:** Saved locked accuracy is **73.07%** and balanced accuracy **73.17%**; training accuracy **79.60%** is not a generalization result. 80% held-out would require roughly +6.93 percentage points on accuracy and +6.83 on balanced accuracy, and may not be feasible for this dataset/protocol.

**Fix:** Create a **separate, explicitly preregistered `research/v2-80-experiments` branch or versioned experiment namespace**, with its own config, budget and candidate registry. Improve **development-only** metrics honestly using grouped folds. Do not touch the already consumed locked holdout or reuse it as a new independent test. To make a new confirmatory 80%-plus claim, acquire a genuinely independent, never-inspected test cohort with comparable labels/features and source rights, after selecting the new method.

**Acceptance:** Produce an honest table showing original nested-development 70.94% balanced accuracy, the new development-only estimate with uncertainty, and clearly labeled *new external* test (if obtained). `80% not achieved` is a perfectly valid final outcome. Do not round 79.x to 80, cherry-pick a seed, tune on original test, adjust labels to inflate accuracy or quietly mix artist groups.

### ISSUE-008 -- Dataset acquisition / codec confounding remains measurable [RESEARCH LIMITATION, P1]

**Evidence:** `results.json` reports a **61.08% balanced-accuracy** development classifier using source-style nuisance features (codec, extractor version, sample rate, duration), versus a 65% predeclared blocking threshold. It did **not** trip the predefined block, but is materially above chance.

**Fix:** Quantify source/codec/duration/extractor distributions by class and group, compare restricted matched subsets on development only, evaluate source-invariant representation or matched-source ablations when statistically feasible, and ensure preprocessing does not key on nuisance columns. This study is about *audio descriptors*, not causation of chart performance.

**Acceptance:** Publish confounder tables and development-only ablation comparisons with uncertainty, not just a pass/fail flag. Do not claim zero bias because the existing threshold was not crossed.

### ISSUE-009 -- Within-group label permutation results are high [RESEARCH LIMITATION, P1]

**Evidence:** Development label-permutation controls were approximately **65.98%, 66.48%, 66.00% balanced accuracy**, not 50%; only 412 mixed-label groups were exchangeable, so this is not a global random-label null. Existing report itself cautions about residual artist/cohort structure.

**Fix:** Investigate carefully (do not call it a code bug without proof): global versus within-group null designs where mathematically valid, matched controls, label imbalance within artists, source and release-era effects, explain what each permutation tests, publish group-level and per-era distributions. Never use permutation outcomes as an excuse to relax artist-connected splits.

**Acceptance:** Report diagnostics and limitations, resolve actual leakage if found, and retain original public result unchanged.

### ISSUE-010 -- Large identity group and incomplete aliases may affect generalization [RESEARCH LIMITATION, P1]

**Evidence:** `results.json` shows 2,633 groups with **largest connected component 636 songs**; grouping links known artists, names, album, work proxies, recording and audio hashes. Unknown aliases/covers can still escape detection. Source review disclosed this.

**Fix:** Test group-size distribution, held-out artist coverage, music-genre and release-era strata if available as **diagnostic metadata only**, compute per-group bootstrap / error analysis, and flag duplicate or alternate recording identities for manual review. Do not add artist or chart metadata as model predictors.

**Acceptance:** `group_leakage_audit.md` includes leakage checks, largest-group sensitivity, diagnostics on development and remaining unavoidable alias limitations.

### ISSUE-011 -- Published HSP-S Class 0 is NOT verified lifetime non-hit [CONFIRMED SCIENTIFIC LIMITATION, P1]

**Evidence:** HSP-S `label()` returns 0 when publisher `peakPos` and `weeks` are both missing, so negatives mean *no publisher-matched Billboard entry in the benchmark*, not proof of no chart appearance anywhere or forever.

**Fix:** Keep accurate classification labels: `Billboard chart entry in HSP-S benchmark` and `No chart entry in HSP-S benchmark`. Add succinct explanations in app, README and final report. Do not rename the task to reliable future hit forecasting; no independent lifetime absence verification is present.

**Acceptance:** All public-facing statements use the correct label scope; no claim of 80% on future unseen commercial music.

### ISSUE-012 -- Experiment title no longer matches the deployed feature method [CONFIRMED COMMUNICATION GAP, P1]

**Evidence:** Root heading and app still say `Chorus Hit Predictor`, but `main` predicts with **436 whole-recording Essentia descriptors**, not 15-second repeated-chorus librosa features. Original chorus project was moved to an archive.

**Fix:** Design accurate branding: e.g. **`Hit Song Prediction | Audio Feature Intelligence`**, with historical project name in an `Origins / Archived Repeated-Chorus Study` section if institution requires it. Explain the real method prominently above the fold, not only in footnotes. Faculty may require approval for methodological change.

**Acceptance:** A first-time visitor cannot mistake the 73.17% HSP-S score for the original repeated-chorus study. App page and README titles agree.

### ISSUE-013 -- The original locked holdout has already been accessed [CONFIRMED PROCESS CONSTRAINT, P0 for experiments]

**Evidence:** `hsp_s_results/results.json` says `test_access: USED_ONCE`; the code writes `test_access.json` *before* opening test labels and refuses reuse. The original test score is now public, making iterative optimization against it especially risky.

**Fix:** Keep original results and artifact immutable. New experiments use the original **development** subset for group CV and a **new independently sourced holdout** for any new confirmatory test. DO NOT delete the access ledger, alter seeds/identity groups or republish the old holdout as fresh. If a new external set cannot be obtained, describe all additional scores as exploratory/development only.

**Acceptance:** Old test is untouched; all new score labels unambiguous; no leakage through README claims or model selection.

### ISSUE-014 -- Candidate warnings require interpretation, not cosmetic removal [RESEARCH LIMITATION, P2]

**Evidence:** Prior run: 486 fit trials complete, 0 failed; **38 trials emitted convergence warnings** (36 stochastic optimizer and 2 Liblinear). Winning final RBF SVM had no warning per handoff.

**Fix:** For v2 experiments, record convergence/solver diagnostics and investigate stability only in new development runs. Do not retroactively modify original result files or hide warnings.

**Acceptance:** Report failed/skipped/warned trials separately; reproduce the original selected model's outputs before expanding experiments.

### ISSUE-015 -- No reproducible current-MP3 accuracy validation [CONFIRMED GAP, P0]

**Evidence:** Existing accuracy was computed on the publisher HSP-S feature table, not from uploaded audio. Uploaded MP3s can be from different encoding, mastering, distribution, genre and Essentia versions.

**Fix:** Require matching recording-version reference pairs to measure feature drift and domain shift, and if possible an **independent labelled MP3 evaluation** before presenting performance claims. No extrapolating 73.17% to any new MP3.

**Acceptance:** Separate model provenance from audio input provenance; publish `MP3 validated / experimental / unavailable` state and actual evidence for any claimed audio accuracy.

## P1 / application UX, release engineering and documentation

### ISSUE-016 -- Main app is functional but limited as a project showcase [ENHANCEMENT, P1]

**Evidence:** `app.py` shows four headline numbers, a JSON expander, a CSV upload box, predictions and a warning. It does not visualize the already stored confusion matrix or model-comparison chart, provide upload tabs, show useful example controls, or explain classes visually.

**Fix:** Build tasteful Streamlit tabs or sections: **Overview**, **Predict (MP3 / CSV)**, **Model Evaluation**, **About / Methodology**. Metrics cards; confusion matrix figure from existing `docs/hit_nonhit/hsp_s_results/confusion_matrix.png`; results explanation; record-count/sample input and download; audio preview/waveform only for authorized local upload; responsive layout and informative validation states. Keep numerical result labels precise.

**Acceptance:** App loads from fresh clone after model setup, both team names/SRNs remain, all saved metrics remain correct, predictions work for real CSV sample, and empty/invalid/unsupported audio conditions are graceful.

### ISSUE-017 -- README is technically good but visually underdeveloped [ENHANCEMENT, P1]

**Evidence:** Root `README.md` has scientific details and setup but no professional hero/banner, CI badge/status badge, prominent methodology graphic, application screenshot or dedicated visual demo section. It can more clearly differentiate the current experiment from the historical title.

**Fix:** Rewrite as a polished, accurate GitHub landing page following Section 5. Keep full run/quickstart commands, team attribution, CC BY 4.0 citation, disclaimers, reproduction, platform constraints, archive link and exact reported performance. Embed existing *real* charts via relative image paths, and screenshot the actual running app if available. Badges may point to genuine GitHub Actions checks; do not invent stars/deployments/models/metrics. GitHub supports relative image links (https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes).

**Acceptance:** README is readable in GitHub browser on mobile and desktop, links work, figures are actual outputs, no accuracy overclaim, quick start works from clean clone.

### ISSUE-018 -- Some archived handoff text contradicts current `main` [CONFIRMED DOC STALENESS, P1]

**Evidence:** `docs/hit_nonhit/hsp_s_results/handoff.md` still describes `pages/1_Hit_vs_NonHit_Features.py` and says the original audio-upload page/model remain intact in the same app. Current repository tree **does not contain `pages/`**; current `app.py` is the HSP-S-only app and the old chorus app is archived. The document header notes a later migration, but some detailed instructions are stale.

**Fix:** Clearly label this handoff as **historical snapshot (10 Oct)** and link to `docs/repository_migration.md`; add an 11 Oct current-state errata or a new up-to-date operational guide. Do not rewrite the historical experiment as though its original state never existed.

**Acceptance:** Every current demo instruction and active path in new README points to files that exist on `main`; users are not told to open missing Streamlit pages.

### ISSUE-019 -- Legacy config constants reference archived paths [CONFIRMED STALE PATHS, P2]

**Evidence:** `chorus_hit/config.py` still defines `DATA_PATH` as `data/chorus_features.csv`, `MODEL_PATH` as `results/v2/v4_std74_001/model.joblib`, plus 518-feature chorus constants. Those files are not in current `main` tree. These constants coexist with necessary `ROOT` and shared imports.

**Fix:** Audit all usages and deprecate or clearly mark historical constants without breaking saved bundle verification, shared imports, archived code or reproduce instructions. Keep current HSP-S source-of-truth config explicit; do not delete transitive dependencies blindly.

**Acceptance:** New/current code does not refer to missing original files, tests are green, and documentation does not direct users to phantom locations.

### ISSUE-020 -- Performance/repeated model verification can be measured and improved [DESIGN RISK, P2]

**Evidence:** `app.py` loads model through `load_model` on each rerun, and `load_model -> verify` hashes all listed model files before deserializing. Current bundle contains 499 archived files. Streamlit reruns on interaction.

**Fix:** Benchmark actual rerun/reclassification latency and memory first. Consider `st.cache_resource` keyed by manifest/code identity for **trusted, unchanged, verified** model, without suppressing integrity checks or accepting untrusted pickle/joblib uploads. Avoid repeated full-hash verification in the hot path once a safe cache invalidation scheme is proven.

**Acceptance:** Same exact predictions and integrity behavior; measured loading/interaction time improves (if previously excessive); no stale/incompatible model accepted after file changes.

### ISSUE-021 -- Limited automated coverage for new upload and current app behavior [CONFIRMED GAP FOR NEW FEATURE, P1]

**Evidence:** Current cleaned project has 26 automated tests; older 285 tests live in archive. Current tests cover CSV/model installation and some safety controls but no genuine MP3 intake or Essentia feature contract.

**Fix:** Expand current tests around file input modes, decoder contract, schema mapping, parity failure, budget/timeouts, cleanup, error messaging, CI optional Essentia environment and snapshot UI. Preserve all 26 existing passes. Optional slow external-audio tests must be clearly separated from synthetic/fake fixtures.

**Acceptance:** New CI passes tests reproducibly on pinned Python 3.12 without requiring copyrighted songs, secrets or gigabyte source downloads. Tests distinguish mocked audio extraction from genuinely verified MP3 accuracy.

### ISSUE-022 -- Dependency/platform support for Essentia needs proof [DESIGN RISK, P1]

**Evidence:** Current pinned `requirements.txt` installs librosa but **not Essentia**. HSP-S source vectors came from Essentia. Project supports Python 3.12 macOS/Linux; Essentia installation varies by OS and package/CLI version.

**Fix:** Research available supported native/CLI/container extractors, pin runtime and extraction profile separately from the existing model's pinned Python dependencies. Test on the user's actual macOS environment and GitHub Actions Linux if feasible. Do not break the existing clean setup just to add a potentially unsupported optional codec library.

**Acceptance:** Explicit installation/feature availability instructions for each platform and tested fallback when Essentia cannot install; CSV-only mode remains functional.

### ISSUE-023 -- Inference score semantics and explainability need better UI [ENHANCEMENT, P2]

**Evidence:** RBF SVM returns **uncalibrated decision margins**, not probability estimates. The current UI labels them correctly but could explain them more clearly, and shows no per-feature interpretability.

**Fix:** Explain `positive score` / `negative score` versus true class labels and uncertainty. Optional grouped, development-only permutation feature-family importance is safer than claiming causal explanations. If calibrated probabilities are desired, learn calibrator using properly grouped development CV, measure Brier/calibration on untouched *new* data, version the model and label scores accurately.

**Acceptance:** No percentage-like likelihood UI unless validated calibration; explanations identified as model associations, not causal songwriting advice.

### ISSUE-024 -- No public playable demo of the current MP3 mode [ENHANCEMENT, P2]

**Evidence:** GitHub README provides `streamlit run app.py` and a CSV example; no public hosted deployment or MP3 test clip is referenced. Audio recording redistribution would need permission.

**Fix:** Optional: add real screenshot and a short recorded walkthrough of the app, kept small and appropriately licensed. Avoid deploying an unvalidated MP3 classifier or leaking raw sample tracks. If hosting, account for 13.65 MB model installation, pins, memory and uploaded-file safety.

**Acceptance:** Fresh clone demo documented, no fake live URL or copyrighted track in public repo.

---

# 2. MP3 + CSV feature implementation design (required)

## 2.1 UI information architecture

A Streamlit interface with a crisp project header and transparent source statement:

- **Overview:** current verified test accuracy 73.07%, balanced accuracy 73.17%, 7,728 records, model `RBF SVM`, dataset `HSP-S`, direct links to methodology/results.
- **Predict | Feature CSV:** working exact 436-feature upload, sample download, template download, preview, validation summary and downloadable results. Show 1-100 row limit, separate 5 MB CSV limit, feature schema help.
- **Predict | Audio (MP3):** permitted local file upload, format/duration/size validation and waveform metadata; extraction status/progress; HSP-S prediction **only once extractor equivalence has passed Section 2.2**; otherwise clear `Experimental audio analysis only - no validated prediction` state.
- **Results & Explain:** saved confusion matrix, class supports, held-out vs development distinction, original methodology caveats, source bias and confidence interval.
- **About:** scope, team names/SRNs, original Eric Liu paper vs current HSP-S method, sources/CC BY, historical archived project, GitHub links.

**Do not make `MP3 upload available` synonymous with `validated MP3 model`.** Every visible classification must be derived from a real trained model receiving a justified input representation.

## 2.2 MP3 extraction correctness decision tree

1. Inventory 436 HSP-S feature keys and source Essentia versions/profiles from pinned source metadata. Determine field extraction algorithms, statistic aggregation and units. Do this BEFORE writing an extractor.
2. Try a pinned Essentia MusicExtractor / streaming extractor with the closest historically equivalent profile on one or two **permitted local test files** supplied by the user or with explicit redistributable test licenses. Confirm the library actually runs in Python 3.12/macOS or through a supported isolated CLI/container.
3. Build explicit mapping `Essentia pool -> required 436 ordered scalar column names`. Reject unmapped fields, missing algorithms, vector-to-scalar assumptions, inconsistent units, overflow and non-finite feature values. Do not fill a missing extractor-derived feature with zeros just to pass the schema.
4. If any identical-master reference files with HSP-S/AcousticBrainz source vectors can legitimately be obtained, compare differences feature-by-feature; predeclare acceptable parity criteria and publish pass/fail as a report. Also check distribution shift of new audio features versus HSP-S training distributions, class/source codec effects and out-of-distribution alerts. Hash references, record version and review outcome.
5. **PASS:** Enable clearly labeled, experimentally validated `MP3 -> whole-recording HSP-S inference` with its own compatibility evidence. Report test evidence for new MP3 predictions separately; never copy over the historical held-out 73.17% as if measured on uploaded audio.
6. **FAIL / NO MATCHED REFERENCES:** Keep MP3 upload for decoding and descriptive audio analysis only, show a **not validated** message, preserve the fully working CSV predictor. If another legitimately labelled audio source is available, a **separate** MP3-compatible model can be built/evaluated without pretending it is the HSP-S one; timebox it.

Do not revert to original 518-feature librosa/pychorus extraction for the deployed HSP-S model. It has a different contract; only a properly trained separate model can consume that representation.

## 2.3 File-handling and security contract

- MP3: validate actual codec/container, not just extension; limit user-supplied compressed and decoded sizes, duration, channels and sampling rate. Default suggestion: 25 MB app-level MP3 cap, <=480 seconds (adjust via measured resource budget). Input is local ephemeral data.
- CSV: <=5 MB, 1-100 records, exact 436 typed columns; missing values only if compatible with the original imputation policy and at least one real descriptor per row; no `Inf`, duplicate names or unexpected metadata/chart columns.
- Accept `.wav` and `.flac` later only after testing and updating the server cap; do not claim every codec works.
- Model upload is **not** supported; reject `.pkl`, `.joblib`, pickle, executable zip and path traversal. Never call `joblib.load` on user file.
- No media downloader, subscription tokens, commercial chart scrape or remote inference API is required to upload permitted local music.
- Temporary file names should be random and isolated. Delete on all exit paths, including raised exceptions. Do not persist audio to repo/CI outputs or logs. Gracefully handle missing optional native extractor and model not installed.
- Keep loading/prediction limits and provenance explicit; do not expose or infer artist/title from unknown MP3 content unless separately reliable.

## 2.4 Suggested un-hashed module architecture

**Avoid new or modified files under `chorus_hit/hsp_s/*.py` until ISSUE-004 is solved.** Suggested paths:

```
app.py
chorus_hit/audio_upload/__init__.py
chorus_hit/audio_upload/validate.py
chorus_hit/audio_upload/extract_essentia.py
chorus_hit/audio_upload/feature_contract.py
chorus_hit/audio_upload/service.py
chorus_hit/ui_components/__init__.py
chorus_hit/ui_components/overview.py
chorus_hit/ui_components/csv_input.py
chorus_hit/ui_components/audio_input.py
chorus_hit/ui_components/results.py
scripts/check_audio_feature_parity.py
tests/test_upload_csv.py
tests/test_upload_audio.py
tests/test_feature_compatibility.py
```

These paths are **suggestions** rather than existing code; adapt to actual application architecture. Preserve `models/bundle.json` and the immutable ZIP SHA; do not move/delete historical `hsp_s/` code whose hash is needed by the model. Add optional native extractor dependency outside core pins if necessary. Use `AppTest` or appropriate verified browser tests.

---

# 3. ML research workstream: pursue >=80% *honestly*

The goal is **at least 80% accuracy if technically achievable and independently validated**. It is a **stretch**, not an entitlement. Current locked accuracy = 73.07%; current nested dev accuracy = 70.97%. The difference to an honest 80% dev estimate is roughly nine percentage points. The existing locked test is consumed and must NOT be reused to optimize or certify a new model.

## 3.1 Freeze the old result and capture baseline

- Record audited commit, SHA of packaged model, 73.07% / 73.17%, 436-feature schema, HSP-S license, and current metrics.
- On a fresh clone, execute Python 3.12 setup, install, read-only four-row prediction, current 26 tests and Streamlit smoke. Capture command/versions/durations.
- Do not overwrite `models/hsp_s_run_001.zip`, `models/bundle.json`, `docs/hit_nonhit/hsp_s_results/results.json`, or previous split/test access records.
- Identify authoritative *development record IDs and groups* for new work. Never inspect previous locked test to choose candidate hyperparameters or feature subsets.

## 3.2 Source and label audit (development only)

- Confirm no label proxies (peakPos, weeks, chart ranks, popularity, streaming counts, release-year metadata, artist IDs) in predictors.
- Quantify class-conditional missingness, feature corruption, duplicate versions, artist-group size, codec/extractor distribution, high-correlation and missing-value patterns.
- Avoid selecting data/removing hard examples based on resulting model errors. Audit missingness in grouped folds; do not fit anything on validation or prior held-out test.
- Publish limitations about HSP-S publisher-matched labels; all features are *whole recording*, not chorus-only.

## 3.3 Structured development-only improvement experiments

Prioritize a **bounded, reproducible experiment matrix**, not thousands of unrecorded ad hoc searches:

**Stage A (cheap, controlled):**
1. Reproduce the development baseline with its original independent artist-connected groups and same feature contract.
2. Compare robust/quantile vs standard scaling (inside fold), capped outlier transformations and median imputation; check *whether apparent improvements are caused by source artifacts*.
3. Pre-register modest feature-family sets (rhythm, tonal, lowlevel, selected spectral summaries), regularized feature selection fitted **inside** folds, and dimensionality reduction.
4. Compare RBF SVM C/gamma grids and supervised threshold optimization using **out-of-fold development predictions only**; keep threshold and probability calibration distinct. No use of old locked test.

**Stage B (if Stage A improves):**
5. Test gradient boosting families (HistGradientBoosting, optional XGBoost / LightGBM / CatBoost where environment and licensing are feasible), ExtraTrees and RandomForest with bounded budgets.
6. Test class-balanced, leakage-safe ensembles or stacking constructed from group out-of-fold predictions only; no stacking on in-sample outputs.
7. Use grouped calibration if probabilities are desired; track Brier/reliability diagram. Never display uncalibrated margins as hit likelihood.
8. Examine per-artist/source/era error and robustness, including nuisance-only and permutation controls. Report group bootstrap intervals, not just single split accuracy.

**Stage C (honest claims):**
9. Lock final candidate based on predeclared development selection; report nested dev performance and comparison to original **development** score, noting multi-experiment selection risk.
10. For a new held-out accuracy claim (especially >=80%), source a truly independent, untouched external benchmark or independent subset with compatible definitions and features **not accessed during method design**. Predeclare eligibility and evaluation once. If no such set exists, report new results as exploratory CV only.
11. Never treat the original HSP-S final test as fresh; never delete its one-use ledger, cherry-pick split seeds, leak labels, increase reported scores by changing the classification target without disclosure, or report training accuracy as target attainment.

**Research budget:** Timebox v2 feature/learner exploration (suggested 3-5 hours compute / bounded number of runs given deadline). Prioritize a functional tested UI and README even if 80% remains out of reach. Record every attempted configuration, failed fit, warning and fit wall time.

## 3.4 Required research report

Write `docs/research_v2/model_improvement_report.md` with:

- Source, license, data IDs, excluded and retained rows, class counts, groups, feature count.
- Old immutable model result as *historical baseline*, and new development-only outcome.
- Grid/trials (including those that underperform), selected candidate and exact hyperparameters.
- Nested grouped CV balanced accuracy/accuracy/precision/recall/F1, per-class confusion matrix, group confidence intervals.
- Label leakage, artist/recording overlap, codec and source nuisance, group-permutation checks.
- Exact implementation/data hashes, random seeds, environment and commands.
- Real new externally held-out metrics **only if independently collected and untouched**; otherwise `NOT AVAILABLE`.
- Decision: `80% genuinely confirmed / >=80% observed on dev only / 80% not reached / incomplete`.
- Honest limitations; zero claims that the old repeated-chorus model reached these results.

---

# 4. Tests, acceptance and delivery gates

## 4.1 Test matrix

**Existing baseline (must remain passing):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
python scripts/setup_model.py
python -m pytest -q
python -m streamlit run app.py
```

Review streamlit smoke on a **new clone**. Test the four-row CSV sample and exact saved labels and uncalibrated scores. If `app.py` starts but the browser UI is not tested, report that accurately.

**New automated tests to implement (with synthetic/generated fixtures and/or explicitly licensable tiny audio):**

1. CSV exact valid schema; wrong order/extra/missing columns; duplicate columns; old 518-feature mismatch with guidance.
2. CSV NaN/Inf/numeric coercion, non-UTF-8, corrupted quoted rows, huge cell, empty and >100 rows; 5 MB rejection.
3. MP3 valid/invalid codec/extension, missing optional Essentia, corrupted/oversize/truncated, duration >cap, zero-byte, silence and unbounded/failed decoding.
4. Audio features 436 required ordered keys, finite values, exact source field mapping, failure on absent/mismatched keys, and distinct extraction profiles.
5. Full MP3 -> prediction **only** when verified parity proof is present; otherwise fail-closed and still display useful UI.
6. Temporary-file and memory cleanups on success and exception; never persist copyrighted audio to tracked paths.
7. Existing model bundle hash/manifest, code hash, installer, one-use test ledger, exact four-row outputs remain unchanged after UI improvements.
8. Streamlit AppTest: both modes, readable validation, metrics, no fake probabilities, screenshot available only if actually captured.
9. CI on Python 3.12 macOS/Linux as feasible; optional extractor unavailable should be tested as a graceful condition, not make CSV CI fail.
10. README relative links/images, current path references, attribution/license and correct historical model boundaries.

**Optional genuine integration tests:** For a permitted 10-30 second audio fixture, exercise actual decoding and Essentia extraction. This is a real extractor software test, **not** a measure of hit-classification accuracy. Pin any redistributed fixture to a verifiable CC/license origin.

## 4.2 Release gates

- **G0 Base preservation:** old bundle SHA and evaluation unchanged, original 26 tests pass.
- **G1 CSV:** sample file and arbitrary schema-valid CSV work; errors are specific and safe.
- **G2 Audio technical:** MP3 selection/decoding/preprocessing tested under caps, no files leaked.
- **G3 Audio scientific:** 436-feature mapping/compatibility independently validated; if not, clearly label preview-only and do not return an invented class prediction.
- **G4 Model research:** development score improvements honestly compared with original dev; old test not re-opened; independent external score only with fresh separate cohort.
- **G5 UX/README:** figures and screenshot real, links work, citation and methodology accurate, app matches README.
- **G6 Delivery:** new checks pass at actual new commit SHA, completion report has true statuses, no merge without authorization.

---

# 5. Root README overhaul specification

The user explicitly wants a **visually professional GitHub front page**, not just a long wall of text. **Preserve truth over marketing**. Update `README.md` with polished Markdown, badges and *actual* repo images. Suggested order:

1. **Hero:** `Billboard Hit Prediction Using Audio Features` or accurate selected title, one-line description; mention historic original `Chorus Hit Predictor` name without falsely implying active chorus extraction. Simple 1-3 sentence problem/solution.
2. **Badges:** a real GitHub Actions CI badge pointing to `.github/workflows/tests.yml`, Python 3.12, HSP-S CC BY 4.0 and Streamlit; avoid fake deployment/download/best-SOTA claims.
3. **At a glance:** genuine **73.07% test accuracy / 73.17% balanced / 7,728 songs / 436 whole-song features / RBF SVM**. Any new >=80% result only after independent verification; label dev vs external test.
4. **Actual app screenshot:** add a verified screenshot of the improved running app under `docs/assets/` with descriptive alt text, sensible image width and no copyrighted album artwork.
5. **Why this problem?** Plain English: classify retrospective Billboard chart association using acoustic descriptors, not future popularity prophecy.
6. **End-to-end workflow diagram:** Mermaid or a verified static diagram: HSP-S feature data -> leakage-safe preprocessing -> group CV -> selected SVM -> evaluated model -> CSV inference. Add conditional MP3 -> Essentia-compatible extractor only once validated.
7. **Prediction modes table:** `CSV: verified with 436 Essentia descriptors`; `MP3: validated or experimental (actual test report link)`. No false green checkmarks for unvalidated audio.
8. **Results:** nested development and locked test table; real confusion matrix and model-comparison PNGs using repository-relative paths (already in `docs/hit_nonhit/hsp_s_results/`). Discuss false positives/false negatives and dataset biases.
9. **Quick start:** Python 3.12 venv, `pip install -r requirements-dev.txt`, `python scripts/setup_model.py`, `python -m streamlit run app.py`, sample CSV. Explain no retrain needed for demo. Optional audio extraction dependencies separate.
10. **Use it:** screenshot/GIF if captured, CSV template, input limits, score margin semantics, where to find model results.
11. **Methodology and reproducibility:** dataset DOI, features and label definitions, 27 candidates, grouped CV, held-out test once. Place detailed commands in docs and link to them.
12. **Tech stack and folder tree:** Streamlit, Python, scikit-learn, pandas, Essentia source features, optional validated audio tool; only actual repository files.
13. **Limitations:** whole-song vs repeated-chorus, publisher no-chart-match vs verified lifetime non-hit, external-domain shift, inherited confounding, future-hit claims excluded.
14. **Authors/team attribution:** Dhanush Sai Suprapadha (PES2UG24CS154), Deepthi V (PES2UG24CS150), original paper and HSP-S authors / DOI / CC BY 4.0, archive repository link.
15. **Tests and support:** real CI badge, `pytest -q` current counts only from latest run, exact model setup and troubleshooting, no unsupported claims about Windows native training.

**Do not** bloat the README with 15 screens of prose; organize into readable sections, short tables, screenshots and links to detailed docs. Use relative links and images; validate URLs, alt text, spacing, no broken `pages/` link. Improve current scientific prose rather than stripping important cautions. No badge or screenshot should be fabricated.

---

# 6. Implementation order and branch policy

The user wants the work done, not another speculative architecture essay.

**Phase 0 -- Audit/backup (first):** Confirm actual HEAD, clean working tree, current README, code hash boundaries and whether a PR already exists. Create a new branch from latest `main`, suggested `feat/mp3-csv-ui-readme-research-v2`; do not reset/force push/merge automatically. Snapshot immutable bundle SHA, expected four-row predictions and current `pytest`/CI.

**Phase 1 -- High-value usability fixes:** Implement polished CSV path, better error messages, app tabs, current results figures and optional real screenshot. Check bundle loads throughout. Fix stale docs/current links. Commit one small tested change.

**Phase 2 -- MP3 feasibility:** Before committing to a native dependency, do a bounded Essentia install/profile/schema audit (suggest 1-2 hours). Then implement safe upload/preview and **only enable classification after compatibility evidence**. If parity cannot be proven, report blocker and keep meaningful preview rather than lying to faculty.

**Phase 3 -- Model experiments:** Start separate, budgeted, predeclared development-only comparisons to pursue 80%. Safeguard the already-consumed holdout. If independent new test cannot be acquired, explain that 80% cannot be honestly confirmed. Do not destabilize working app for speculative accuracy.

**Phase 4 -- README, docs and final verification:** Refine README with screenshots, diagrams, real result table, setup instructions, citations and a feature-mode compatibility table. Run tests and CI, verify links/current paths, build report and PR.

**Phase 5 -- Publish:** Push verified commits, open a new PR if permitted by user, **do not merge**. Keep original results and archive preserved. No copyrighted audio or private paths in public GitHub. Present final handoff in Codex conversation so user can paste it into ChatGPT for review.

If there are no permitted local audio fixtures or no compatible extractor, Codex should implement the CSV/UI/docs/testing work and explain **one precise input or approval** needed for an MP3 validation experiment. Do not leave everything unfinished waiting for audio.

---

# 7. Required Codex outputs

Commit or prepare these **new** artifacts (adjust paths as necessary, not fake status):

- `docs/audit/CODEX_REPO_AUDIT_MP3_CSV_80_PERCENT_README.md` -- copy of this plan or link to it, with current SHA reconciliation.
- `docs/audit/issue_resolution_matrix.md` -- one row per ISSUE-001..024: `CONFIRMED / FIXED / NOT REPRODUCED / PARTIAL / BLOCKED / DEFERRED`, evidence, commit, test.
- `docs/audio_upload/feature_compatibility.md` -- pinned 436-descriptor MP3 extractor feasibility/parity report.
- `docs/audio_upload/upload_contract.md` -- size/format/security/schema limits, supported modes and examples.
- `docs/research_v2/model_improvement_report.md` -- all candidate results and accurate dev/external-test distinction, even if target misses 80%.
- `docs/audit/final_delivery_report.md` -- exact completion, commands, CI status, known limitations and remaining inputs.
- Updated `README.md`, `app.py`, supporting modules and tests, plus genuinely captured screenshots/figures.

**Always finish with this user-facing summary format:**

```
CURRENT BRANCH / HEAD:
BASELINE PROTECTED: model hash, original 73.07% / 73.17%, original test access status
CURRENT CI: workflow URL, actual passed/failed tests
MP3: supported and validated / preview only / blocked (include parity evidence)
CSV: supported / errors / count limit / sample result
STREAMLIT: what changed, exact run command, screenshots if real
ACCURACY WORK: new development outcome, evidence, NOT AVAILABLE for new untouched test unless one exists
README: updated sections and working links
ISSUES: fixed / partial / blocked / deferred by ID
FILES / COMMITS: exact paths and commit URLs
ONE REQUIRED USER ACTION: only if there is a concrete blocker
```

**Do not claim all issues are fixed when only a subset is tested. Do not fake or guarantee 80%. Do not change source/license/metric semantics for marketing. Keep the repo functional at every checkpoint.**

---

# 8. Files Codex must inspect before proposing a fix

- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/app.py
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/.streamlit/config.toml
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/chorus_hit/hsp_s/data.py
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/chorus_hit/hsp_s/train.py
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/chorus_hit/hsp_s/evaluate.py
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/chorus_hit/hsp_s/modeling.py
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/chorus_hit/config.py
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/scripts/setup_model.py
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/models/bundle.json
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/README.md
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/docs/hit_nonhit/hsp_s_protocol.md
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/docs/hit_nonhit/hsp_s_results/results.json
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/docs/hit_nonhit/hsp_s_results/handoff.md
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/docs/repository_migration.md
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/tests/test_current_project.py
- https://github.com/DhanushSS/chorus-hit-predictor/blob/main/tests/hit_nonhit/test_hsp_s.py

**Useful external references:**
- HSP-S source and authors: https://zenodo.org/records/5383858
- Essentia MusicExtractor: https://essentia.upf.edu/tutorial_extractors_musicextractor.html
- AcousticBrainz history/technical limits: https://musicbrainz.wordpress.com/2022/02/16/acousticbrainz-making-a-hard-decision-to-end-the-project/
- GitHub relative images/README: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes

# END -- Start with Phase 0 and then implement. Do not stop at commenting on this document.
