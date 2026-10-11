# Repository cleanup and archive — 11 October 2026

## Requested change

The team requested a separate copy of the old repository and a current, professional
project on the original repository's default `main` branch. This supersedes the
earlier instruction to keep the new work exclusively on the feature branch.

## Archive verification

Public archive: https://github.com/DhanushSS/chorus-hit-predictor-archive-2026-10-11

Before any tracked files were removed, all eight remote branch heads and their
reachable Git history were copied into the archive. The source and archive
`refs/heads/*` and `refs/tags/*` listings matched byte-for-byte. The mirror passed
`git fsck --full`. This preserves the old datasets, models, completed results,
metadata lists and notices as they existed on each branch. `main` in the archive
retains the previous default project; `feat/hit-vs-nonhit-v1` retains all later
work through commit `0c320312415ca9d9f6f89f989eddd21f850ac135`.

A separate local bundle also preserves local Git branches. Local-only artifacts
were backed up separately, including the current trained model and all 54 files
from the original preservation manifest. All 54 archived file hashes matched.
Local-only source downloads, private paths and credentials were not uploaded.
The Git archive copies repository contents/history, not GitHub issues, PR threads,
stars, permissions or workflow-run records. The original repository retains those.

## Current repository

The current checkout removes 145 obsolete tracked files from its latest tree:
old chorus datasets/models, historic experiment outputs, superseded configurations,
old interfaces and their tests. Their contents remain recoverable in the archive
and existing Git history; history was not rewritten and no force push was used.

The retained project contains:

- The original HSP-S implementation, fixed search configuration and saved results.
- Shared code that the immutable training implementation still imports.
- A single current Streamlit app, headed by both team names and SRNs.
- A checksum-pinned model archive, installation script and four-row feature demo.
- Current regression tests, CI, concise README and source/license attribution.

The retained shared modules include some historical names and audio helpers because
of transitive imports. They are dependencies, not additional deployed models.
The HSP-S training and prediction implementation hashes remain identical to those
in the evaluated model. Model packaging changes neither training nor metrics.

## Model delivery and verification

The evaluated run is bundled as `models/hsp_s_run_001.zip` (499 files, 13,654,495
bytes). Its pinned SHA-256 is in `models/bundle.json`. Setup verifies the archive,
rejects unsafe paths, checks the original manifest and environment, and installs
without overwriting another run. The app loads only that evaluated model.

Local checks on the cleaned project:

- `python scripts/setup_model.py`: complete, idempotent installation.
- `python -m pytest -q`: **26 passed** (1.97 seconds).
- Existing 17 HSP-S regressions retained, plus nine current delivery/runtime/app tests.
- Sample CSV predictions match the original model's four reference outputs.
- App tests verify both SRNs, saved metrics, input limitations and missing-model guidance.
- The new installer is tested against corruption, traversal and overwrite attempts.
- Supervised fit timeout, version/platform guards and unavailable intervals remain tested.
- `git diff --check`: passed.

The prior 285-test count applies to the full pre-cleanup project. Retired pipeline
and dataset tests remain in the archive; that count is not presented as the new
suite size. CI on the current project checks setup and all current tests. View
[the workflow](https://github.com/DhanushSS/chorus-hit-predictor/actions/workflows/tests.yml)
for the exact commit-specific outcome.

No fresh model search or held-out evaluation was performed during cleanup.
Balanced accuracy remains **73.17%** for the HSP-S test, with the original caveats.
