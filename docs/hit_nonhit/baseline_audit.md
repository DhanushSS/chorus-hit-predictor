# Phase 0: preserved baseline

Audited 2026-10-08. Clean `main` and fetched `origin/main` both pointed to
`853e1b88fbc6a10198bda9eff1529766f2201d83`. The plan's observed `5524012` was older.
Created `feat/hit-vs-nonhit-v1` from the current main. GitHub reports `isPrivate: false`.
No repository instruction file was found in this workspace.

## Protected state

- Active configuration: `configs/active_run.json`, run `v4_std74_001` (public name: Chorus Hit Predictor).
- 751 inherited recordings, 366 year-end hits and 385 other weekly-chart songs. These are **not** hit/non-hit labels.
- 518 input features; the saved logistic pipeline selects 74 standard deviations.
- Preserved nested development accuracy 54.6063651591%; balanced accuracy 54.6602268898%.
- Historical split: 597 development / 154 previously inspected examples. It is not reused for this research target.
- [preservation_manifest.json](preservation_manifest.json) records SHA-256 for 54 original data, configuration, result and local PDF/presentation files. Ignored local deliverables are checked locally; CI checks the tracked assets.
- `docs/task_alignment.md` covers approval of the older adaptation. **Faculty approval of this new target remains pending.**

## Actual baseline checks

Python 3.12.14, macOS 27.0.1 arm64; existing environment matched every pinned dependency.

| Command | Actual result |
|---|---|
| `.venv/bin/python -m pip check` | No broken requirements |
| `.venv/bin/python -m pytest -q` | 148 passed, 4 existing audio-decoder deprecation warnings, 87.79 seconds |
| `.venv/bin/python -m chorus_hit.readiness` | software_ready=true; existing run valid |
| `.venv/bin/python -m chorus_hit.artifacts` | existing active bundle validated |
| `.venv/bin/python -m chorus_hit.predict --run-id v4_std74_001 --track-id CH0001` | Successful legacy prediction; explicitly a **training example**, not a new test |

Logs: [verification](verification/). The supplied plan's 157-test count does not describe this checkout.
The initial shell could not find `gh` on PATH; `/opt/homebrew/bin/gh` worked. No authentication changes were made.

## README inspection

README correctly described the current single saved model and old labels. Its statement that
accuracy experimentation had stopped referred to the old task; a separately labeled new research
section now describes this requested branch. No historical score, label or config was revised.
No original source recordings, licensed weekly chart export or verified negative cohort was found
in the current dataset directory. Readiness confirms zero independently verified legacy labels.
