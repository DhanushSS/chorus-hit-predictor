# Remediation status — 2026-10-05

Base: `d06c834221c0be688a9ddeb663af82a6dab6136f` (current remote main at inspection).
The initial local branch was `2a67e7f`; it was clean. The three intervening changes
were README claims, the repeated-split script and its existing CSV. Work uses
`fix/audit-remediation`, without resets, deletions of evidence or history rewriting.
Public visibility was verified using `gh repo view --json visibility`.

This is an implementation record, not an accuracy improvement claim. A fixed
item has code/documentation and executed tests. Data limitations remain blocked
where their software/documentation portions are implemented.

| ID | Status | Changed files | Verification / remaining boundary |
| --- | --- | --- | --- |
| C01 | FIXED_AND_TESTED | `chorus_hit/evaluation.py`; `tests/test_validation_regressions.py` | Original 1D exact numeric binary inputs validated before casting; fractional, nonfinite, bool, string, shape and score rejection; scikit-learn agreement and zero-division behavior. |
| C02 | FIXED_AND_TESTED | `evaluation.py`, `train_v2.py`, `app.py`; validation and app tests | Positive integer repeats, aligned non-null groups, class support, deterministic seeds, unavailable intervals with reasons; caller/UI guards. |
| C03 | FIXED_AND_TESTED | `ingestion.py`; validation tests | Shared staged eligibility and strict validation; evidence URLs, canonical ID types, chart window/type, rights, file/container/hash, segment boundary, extractor checks. References are not independently authenticated. |
| C04 | FIXED_AND_TESTED | `data.py`, `artifacts.py`; validation tests | Finite features required at loading and prediction; invalid missing/empty cases fail before training. Valid temporary data fit/save/reload/predict works. Original imputers retained. |
| C05 | FIXED_AND_TESTED | `contracts.py`, `artifacts.py`, `evaluate_run.py`; `tests/test_semantic_artifacts.py` | Every original run loads. Rehashed label/group/fold/winner/membership/count/support/confusion/status mutations fail; source/frozen split, nested procedure folds/trials, target semantics checked. |
| C06 | FIXED_AND_TESTED | `app.py`; `tests/test_app_resilience.py` | Optional V2 loss does not disable baseline/Chorus Hit Predictor prediction. Missing baseline disables historical demo gracefully. Selected-model corruption stops prediction. Methods use selected run. |
| C07 | FIXED_AND_TESTED | `train_v2.py`, `supervision.py`; `tests/test_runner.py` | Actual fit start/runtime supervision, queued-time exclusion, later-fit timeout, checkpoint reuse, changed-task rejection, worker exception/interruption cleanup. Explicit per-invocation resume budget. |
| C08 | FIXED_AND_TESTED | `supervision.py`, `train_v2.py`, README; runner tests | Early non-POSIX failure; bounded POSIX process-group cleanup tested. No Windows training support claimed. |
| C09 | FIXED_AND_TESTED | `scripts/benchmark_models.py`, `diagnostics.py`; `tests/test_diagnostics.py` | Import has no training/writes; callable and guarded CLI; new outputs only; frozen split check and config/code/environment identity. Explicit CLI run into a new directory passed. |
| C10 | FIXED_AND_TESTED | `scripts/leakage_demo.py`, `diagnostics.py`, README; diagnostic tests | Explicit settings/output/budget, bounded workers, warning/failure records, unrounded values and all-data disclosure. Synthetic diagnostic tests only; no new all-751-song search. |
| R01 | FIXED_AND_TESTED | `docs/task_alignment.md`; readiness tests | User confirmed “Faculty has approved this adaptation” on 2026-10-05. Recorded as user-reported approval, not independently received faculty correspondence or exact paper reproduction. |
| R02 | BLOCKED_DATA | `readiness.py`, `docs/evidence_audit.json`, `docs/recording_review_template.json`; readiness tests | Reproducible audit: 751 inherited, 0 independently verified with references, 0 disputed, 0 unknown. Actual external chart review remains unavailable. No labels changed. |
| R03 | BLOCKED_DATA | README, evidence audit, `docs/audio_contract.md` | All 751 recording identities unresolved; no canonical-ID/duplicate resolution claimed. Existing connected-group and separation tests retained. New evidence and versioned regrouping needed. |
| R04 | BLOCKED_DATA | `audio.py`, `app.py`, `docs/audio_contract.md`; audio/schema tests | Machine/user-visible schema versus waveform-parity status; legacy extractor unchanged. Source waveforms/reference feature values unavailable; synthetic tests cannot establish parity. |
| R05 | BLOCKED_DATA | `audio.py`, audio contract; `tests/test_audio_v2.py` | Manual/repeated/fallback metadata, similarity, boundaries, nonfinite/silence/short cases tested. Real annotated chorus review denominator is 0; musical selector accuracy unmeasured. |
| R06 | FIXED_AND_TESTED | `docs/evaluation_registry.json`, README; readiness tests | Original memberships and hashes recorded, including ignored local threshold/historical work and subsequent all-data diagnostic. No fresh-test claim, seed-based independence claim or promotion. |
| D01 | FIXED_AND_TESTED | README, `app.py`, diagnostic descriptions | Universal ceiling claims removed; conclusions bounded to evaluated settings. Original scores retained. Local benchmark is not Eric Liu's runtime or a matched paper comparison. |
| D02 | FIXED_AND_TESTED | README, `docs/audio_contract.md` | Setup, offline dependencies, platform matrix, audit/prediction, new V1/Chorus Hit Predictor paths, partial resume, architecture, limits, troubleshooting; CLI help and representative commands executed. Full studies not needlessly rerun. |
| T01 | FIXED_AND_TESTED | New validation, semantic-artifact, app, diagnostic, readiness tests; expanded audio/runner tests | Negative tests, hash-consistent semantic corruption, predictions through every run, format decoding, synthetic extraction, cleanup and preservation; existing assertions retained. |
| T02 | HUMAN_REQUIRED | `readiness.py`, CI workflow, `docs/demo_checklist.md`; readiness tests | Local pinned environment and headless startup/HTTP health tested. Readiness detects missing/mismatched packages. Actual review-device and permitted-song live rehearsal still pending. |
| X01 | DEFERRED_OPTIONAL | Audio contract / task alignment | No new approved suitable recording collection. Audio preparation is explicitly not a compatible connected-group training path. |
| X02 | DEFERRED_OPTIONAL | This status record | Existing optional embedding cache is not enabled for real experiments. Full reviewed-weight/processor/cache identity hardening remains required before that extension. |

## Regression evidence

Before edits, the actual modules reproduced: fractional labels giving perfect
metrics; zero bootstrap repeats raising `IndexError`; bogus audio metadata marked
eligible by coverage but rejected by strict validation. The initial suite passed
28 tests with four warnings.

An early new-test run caught a mechanical import edit error, repaired immediately.
A cleanup test initially patched Python's shared `time.sleep`, affecting the pool's
own cleanup; the test now injects interruption only at the runner's polling hook
and verifies there are no surviving child processes. Neither failure was hidden
by deleting or relaxing an assertion. Intermediate complete suites passed 134 and
137 tests. The definitive local run passed **144 tests, 4 warnings in 133.76 seconds**,
with no failures or skips. Warnings are the existing `aifc`, `audioop`, `sunau`
deprecations and librosa audioread fallback deprecation; dependency pins were not changed.
The separate readiness tests passed 4/4, including execution without site packages.
`pip check` reported no broken requirements. Python was 3.12.14 on macOS.
Linux CI is requested on the pull request; its live result is separate from this local record.

Exact commands executed from the repository root (the environment's `python` is
`.venv/bin/python`):

```bash
git status --short
git rev-parse HEAD
git fetch origin
git diff --stat HEAD..origin/main
git switch -c fix/audit-remediation origin/main
gh repo view --json visibility
.venv/bin/python --version
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_validation_regressions.py tests/test_semantic_artifacts.py tests/test_runner.py
.venv/bin/python -m pytest -q tests/test_validation_regressions.py tests/test_semantic_artifacts.py
.venv/bin/python -m pytest -q tests/test_runner.py tests/test_diagnostics.py
.venv/bin/python -m pytest -q tests/test_readiness.py
.venv/bin/python -m chorus_hit.artifacts
.venv/bin/python -m chorus_hit.predict --run-id v4_std74_001 --track-id CH0001
.venv/bin/python -m chorus_hit.audit --out output/audits/codex_audit_001
.venv/bin/python -m chorus_hit.readiness --out output/remediation/readiness_initial.json
.venv/bin/python -m chorus_hit.readiness --out output/remediation/readiness_final.json
.venv/bin/python -m chorus_hit.readiness --evidence-only --out docs/evidence_audit.json
.venv/bin/python -m chorus_hit.train --help
.venv/bin/python -m chorus_hit.train_v2 --help
.venv/bin/python -m scripts.benchmark_models --help
.venv/bin/python -m scripts.leakage_demo --help
.venv/bin/python -m scripts.benchmark_models --out output/benchmarks/remediation_check_001 --timeout-seconds 120
.venv/bin/python -m streamlit run app.py --server.headless true --server.port 8502
git diff --check
```

The server returned HTTP 200 and `ok` from `/_stcore/health`; the temporary server
was then stopped. AppTest exercises actual app rendering/buttons, separate from
that server-health check. `CH0001` correctly reported `training_example`, not held
out. The benchmark reproduced exploratory fixed-pipeline accuracy (56.6%, 50.6%,
51.1%); it does not replace Chorus Hit Predictor's 54.61% nested assessment. Detailed local logs are
in ignored `output/remediation/` and the benchmark's new output directory.

## Preservation and scientific semantics

105 tracked data/result/model/active-configuration files match their initial SHA-256
snapshot. 17 pre-existing local experiment/benchmark evidence files also retain
matching hashes. No original result, model, label, feature, grouping, threshold,
candidate hyperparameter, metric definition or active-run configuration changed.
The active run is `v1_baseline`; Chorus Hit Predictor remains selectable research.

Robustness changes reject invalid inputs and artifacts, enforce future execution
budgets, isolate optional UI dependencies and version diagnostic outputs. They do
not retrospectively alter valid stored metrics or imply old Chorus Hit Predictor exceeded its budget.
New reproductions use new run/code identities. Metadata now records extraction
verification boundaries explicitly. No original source audio was obtained, no new
real-song accuracy was established, and no PDFs/slides were produced.

Keep the repository public. Faculty adaptation approval is recorded as reported
by the user. Remaining completion boundaries: independent chart evidence,
resolved identities/duplicates, source waveform parity, annotated chorus checks,
a genuinely fresh collection, and actual review-device rehearsal.

## Final follow-up — 2026-10-06

Rehearsal, Q&A preparation and further faculty/access confirmation are deferred
by the user and removed from the current work list. T02's actual-device step is
therefore deferred, not completed. R01's already-recorded user-reported approval
remains valid. PDFs and slides were not created or edited.

A manual-start validation defect was reproduced: `start_seconds=-1e-9` was
rounded to sample zero and accepted. `chorus_hit/audio.py` now rejects negative
seconds before rounding; the existing boundary regression test includes this
case. This software fix does not change saved-model metrics or extractor settings.

One fixed, predeclared equal-weight ensemble of the eight Chorus Hit Predictor std74 logistic
settings was tested against the fixed Chorus Hit Predictor control. Both used the same 597
development rows and five artist-name-grouped folds (seed 44), with fold-local
preprocessing, native control predictions, ensemble cutoff 0.5 (ties class 0),
40 fits, one CPU thread and a 120-second whole-command budget. The protocol and
code/data/config/fold identities were saved before the first fit. No weights,
thresholds, seeds or member subsets were selected from the results. The 154
historical rows were excluded from all new fits and scoring.

| Metric | Fixed Chorus Hit Predictor control | Fixed ensemble |
| --- | ---: | ---: |
| Accuracy | 56.62% | 55.28% |
| Balanced accuracy | 56.72% | 55.38% |
| Precision | 54.86% | 53.61% |
| Recall | 60.34% | 58.97% |
| F1 | 57.47% | 56.16% |
| Correct predictions / 597 | 338 | 330 |

The ensemble's paired accuracy difference was -1.34 percentage points; its
2,000-resample whole-group 95% bootstrap interval was [-3.82, +0.84] points.
There is no demonstrated improvement. This is reused development data, and the
interval does not account for prior model selection or dependent folds. These
fixed-pipeline scores do not replace Chorus Hit Predictor's 54.61% nested procedure assessment.
The ensemble was not promoted; accuracy experimentation is stopped.

Verification completed:

- The control's track IDs, labels, folds, artist groups and predictions exactly
  match the prior fixed-pipeline benchmark. All 40 fits completed without warnings.
- Recomputed both sets of metrics from the new saved predictions; verified their
  hashes and the pre-run protocol hash. All 105 tracked original artifacts and
  17 earlier local evidence files match their preservation snapshots.
- Focused checks: **18 passed in 4.38 seconds**. They cover historical exclusion,
  fixed members, training/validation separation, one prediction per row, equal
  weights/ties, invalid probabilities, deterministic paired bootstrap, protocol
  persistence before fitting, overwrite refusal, import safety and negative starts.
- Complete local suite: **157 passed, 4 existing deprecation warnings in 138.42
  seconds**; no failures or skips. `pip check` found no broken requirements.
- Public GitHub visibility was verified. The active run remains `v1_baseline`.
  GitHub CI for this follow-up is separate from the local test result above.

Exact executed test/experiment commands:

```bash
.venv/bin/python -m pytest -q tests/test_final_accuracy_check.py tests/test_audio_v2.py::test_manual_boundaries tests/test_diagnostics.py::test_imports_have_no_training_writes_or_warning_filter_effects
.venv/bin/python -m scripts.final_accuracy_check --out output/experiments/final_ensemble_2026-10-06_001
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
git diff --check
```

Full local evidence is in `output/experiments/final_ensemble_2026-10-06_001/`.
The experiment script and exposure registry retain its method, metrics and hashes
without adding generated reports or slides to the repository.
