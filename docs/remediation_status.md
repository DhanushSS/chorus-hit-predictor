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
| C06 | FIXED_AND_TESTED | `app.py`; `tests/test_app_resilience.py` | Optional V2 loss does not disable baseline/V4 prediction. Missing baseline disables historical demo gracefully. Selected-model corruption stops prediction. Methods use selected run. |
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
| D02 | FIXED_AND_TESTED | README, `docs/audio_contract.md` | Setup, offline dependencies, platform matrix, audit/prediction, new V1/V4 paths, partial resume, architecture, limits, troubleshooting; CLI help and representative commands executed. Full studies not needlessly rerun. |
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
51.1%); it does not replace V4's 54.61% nested assessment. Detailed local logs are
in ignored `output/remediation/` and the benchmark's new output directory.

## Preservation and scientific semantics

105 tracked data/result/model/active-configuration files match their initial SHA-256
snapshot. 17 pre-existing local experiment/benchmark evidence files also retain
matching hashes. No original result, model, label, feature, grouping, threshold,
candidate hyperparameter, metric definition or active-run configuration changed.
The active run is `v1_baseline`; V4 remains selectable research.

Robustness changes reject invalid inputs and artifacts, enforce future execution
budgets, isolate optional UI dependencies and version diagnostic outputs. They do
not retrospectively alter valid stored metrics or imply old V4 exceeded its budget.
New reproductions use new run/code identities. Metadata now records extraction
verification boundaries explicitly. No original source audio was obtained, no new
real-song accuracy was established, and no PDFs/slides were produced.

Keep the repository public. Faculty adaptation approval is recorded as reported
by the user. Remaining completion boundaries: independent chart evidence,
resolved identities/duplicates, source waveform parity, annotated chorus checks,
a genuinely fresh collection, and actual review-device rehearsal.
