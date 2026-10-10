# 278-song execution — completion status

**RUN STATUS: BLOCKED_AUDIO**

10 October 2026. All presently unblocked acquisition-workflow changes and checks
were completed. No genuine new-target classifier was trained or evaluated.
The original model remains available in Streamlit.

## Repository and evidence identity

- Public repository: [DhanushSS/chorus-hit-predictor](https://github.com/DhanushSS/chorus-hit-predictor).
- Branch: `feat/hit-vs-nonhit-v1`; main was not changed.
- Starting HEAD: `e1f18b583b1ecf8d04a79eeacdf7ffbe30647736`.
- Tested implementation HEAD: `fa35fe4b3e1d0c6b1f6ef1c4aba8afe44f0e785e`.
  The local test run used the identical code subsequently committed at this SHA.
- Final report is a documentation follow-up to that implementation commit; resolve
  its own SHA through this file's GitHub history or `git log -1`.
- Main inspected at `853e1b88fbc6a10198bda9eff1529766f2201d83`.
- [PR #6](https://github.com/DhanushSS/chorus-hit-predictor/pull/6): **CLOSED, UNMERGED**.
  No reopen, replacement PR or merge was performed.
- All **54/54 protected-file hashes** match the preservation manifest. The fixed
  metadata intake and academic model configuration hashes also match the starting audit.

## Phase status

| Phase | Status | Evidence / blocker |
| --- | --- | --- |
| 0: checkout, metadata, environment, preservation | COMPLETE | Clean starting branch; 278 unique IDs; 139/139; pinned Python 3.12; 54 hashes match |
| 1: bounded audio/source investigation | COMPLETE, AUDIO BLOCKED | Gate recorded after 41.25 minutes; zero authorized exact files; sources and alternatives documented |
| Acquisition workflow improvements | FIXED AND TESTED | Fixed-cohort/label guard, finite chorus positions, local ledgers, rate-limited exact-ID feature probes, privacy ignore rules |
| 2: real chorus pilot | NOT RUN — NO AUDIO | 0 permitted recordings; no detector comparison or human reviews |
| 3: real shared-feature cohort | BLOCKED | 0 genuine 518-feature rows; no legacy vectors reused |
| 4: freeze/train/locked evaluation | BLOCKED | 0 real models; real locked test NOT_USED; 27 configurations preserved |
| 5: application and software checks | VERIFIED WITH LIMITS | 268 tests, synthetic smoke, app route and local HTTP health pass; no real new-target prediction claim |
| Feature-based fallback | RESEARCHED, NOT TRAINED | HSP-S recommended subject to the plan's separate-experiment scope approval |

## Counts and required results

| Required field | Actual value |
| --- | --- |
| Selected metadata records | **278: 139 hits / 139 non-charting candidates** |
| Metadata diversity | 36 lead artists, 67 albums, 278 works, 34 connected metadata groups |
| Authorized audio files obtained and processed | **0: 0/0** |
| Missing / rejected / wrong-version files | **278 / 0 / 0**; no real files attempted |
| Real chorus-feature records | **0: 0/0** |
| Human-reviewed choruses | **0/0** |
| Complete both-class audio match sets | **0** |
| Real connected groups / frozen split support | **0 / NOT AVAILABLE** |
| Metadata-only preflight | 27 development groups (228 records), 7 held-out groups (50 records); not a frozen real test |
| Real dataset fingerprint | NOT AVAILABLE |
| Extractor/config fingerprints | Listed in [real-audio results](real_audio_results_278.md) and machine-readable evidence |
| Models evaluated / skipped | **0 / all 27 blocked before fitting** |
| Best model and configuration | NOT AVAILABLE |
| Training and nested-validation accuracy, balanced accuracy, precision, recall, F1 | NOT AVAILABLE |
| Real locked-test access | **NOT_USED** |
| Locked size / class support | NOT AVAILABLE |
| Locked accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, AP | NOT AVAILABLE |
| Confusion matrix (TN, FP, FN, TP) | NOT AVAILABLE |
| 95% group confidence interval | NOT AVAILABLE |
| Source/codec/permutation nuisance diagnostics | NOT RUN ON REAL AUDIO |
| Acquisition candidate features | AcousticBrainz has submissions for 263 exact IDs: 129 hits / 134 controls; not compatible chorus rows |

No zero-valued accuracy, fabricated confusion matrix, copied historical score or
synthetic score is substituted for missing research results.

## Implemented changes

| File | Change |
| --- | --- |
| `chorus_hit/hit_nonhit/acquisition.py` | Immutable 278-ID/label scope; blank local acquisition ledger; count-only AcousticBrainz batch probe; unknown/error/missing distinction; stops on rate limiting; exclusive output directories |
| `chorus_hit/hit_nonhit/intake.py` | Enforces approved cohort before archive/audio access; rejects NaN, infinity and negative chorus starts; corrects matching comment |
| `.gitignore` | Additional raw audio formats and filled-intake/acquisition-ledger CSVs kept local |
| `tests/hit_nonhit/test_acquisition.py` | 14 tests including parameterized cases: expansion/label/duplicate rejection, changed cohort, non-overwrite, no invented review, malformed/merged/count responses, rate-limit stop, no feature/audio count inflation, privacy ignores |
| `tests/hit_nonhit/test_academic.py` | Added invalid chorus-position cases to the existing intake regression |
| `docs/hit_nonhit/academic_workflow.md` | Documents fixed scope and reproducible ledger/availability commands |
| `docs/hit_nonhit/audio_feasibility_278.md` | Source/permission findings, actual coverage, inspected fallback and exact decision gate |
| `docs/hit_nonhit/chorus_pilot_278.md` | Explicit unrun real pilot and unchanged review/extraction procedure |
| `docs/hit_nonhit/real_audio_results_278.md` | Explicit unavailable real metrics, lineage and limits |
| `docs/hit_nonhit/acquisition_evidence_278.json` | Safe counts, hashes and checks; no local paths, audio or raw third-party features |
| `docs/hit_nonhit/final_48h_status_278.md` | This completion report |

No dependencies, feature columns, training candidates, split seed, test-access
policy, deployed model, original features, historical outputs or prior completion
report were changed. `academic_progress.json` remains the original truthful snapshot.

## Exact checks executed

Commands used `.venv/bin/python` in the repository's pinned Python 3.12 environment.
Local logs are under ignored `output/hit_nonhit/real_audio_48h_20261010/`.

| Command/check | Result |
| --- | --- |
| `python -m pip check` | PASS, no broken requirements (before and after work) |
| `python -m chorus_hit.readiness` | PASS (before and after work); original deployment ready |
| `python -m pytest tests/hit_nonhit -q` before changes | **106 passed**, 69.99 seconds |
| `python -m pytest tests/hit_nonhit/test_acquisition.py tests/hit_nonhit/test_academic.py -q` | **49 passed**, 4.08 seconds |
| `python -m pytest -q` | **268 passed, 4 warnings**, 174.24 seconds |
| `python -m chorus_hit.hit_nonhit.smoke --out output/hit_nonhit/real_audio_48h_20261010/synthetic_smoke` | PASS; `synthetic: true`, `research_accuracy: null` |
| `python -m chorus_hit.hit_nonhit.acquisition init --out output/hit_nonhit/real_audio_48h_20261010/intake` | 278 missing/unknown ledger rows; no invented permissions |
| `python -m chorus_hit.hit_nonhit.acquisition probe-features --out output/hit_nonhit/real_audio_48h_20261010/feature_probe` | All 278 queried; 263 present / 15 no submission; no failures |
| `python -m chorus_hit.hit_nonhit.intake --prepared output/hit_nonhit/academic_prepared_final --discovered data/hit_nonhit_v1/academic --intake data/hit_nonhit_v1/audio_intake.csv --audio-root output/hit_nonhit/real_audio_48h_20261010 --dry-run` | 278 rows, zero ready, 278 missing paths |
| `python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8513 --server.headless=true --browser.gatherUsageStats=false` | Started isolated local process; health 200/`ok`, main page 200; stopped only this check's process |
| Streamlit `AppTest` in full suite | Full app renders without exception; new route says dataset/model not ready; original model/title retained |
| Protected-manifest SHA-256 audit | 54 checked; zero mismatches |
| `git diff --check` | PASS |

Four warnings come from the existing legacy audio fallback: Python 3.13 removals
for `aifc`/`audioop`/`sunau` and librosa's deprecated audioread loader. They did not
fail tests; this branch still uses pinned Python 3.12. The new extractor uses
soundfile and does not introduce that fallback.

## GitHub checks

**PASS** on publication commit `047163c27147fe2a4f72678f2878e736098a8a7d`:
[GitHub Actions run 38024787620](https://github.com/DhanushSS/chorus-hit-predictor/actions/runs/38024787620),
completed 10 October 2026 at 04:47:06 UTC. Linux CI reports **268 passed, four
warnings in 393.12 seconds**. Dependency checks, original-model readiness and
synthetic end-to-end smoke also passed; `research_accuracy` remained null.

The run was explicitly dispatched because push CI covers only `main`/`audit/**`
and PR #6 is closed. The PR was not reopened. This final documentation-only CI
receipt follows that tested publication SHA; executable files are unchanged, and
the receipt itself does not trigger another push workflow. The verified result
belongs to the exact SHA above, not to an earlier baseline or an untested model.

## Remaining blockers and next action

**For the exact-278 repeated-chorus model:** no qualifying permitted source for the
recordings was identified. A source/provider grant covering download and local
analysis plus exact recording identities is required. A permitted batch delivery
is acceptable; the user does not need to download 278 files manually. Human chorus
reviews and sufficient independent groups are subsequent gates, not completed work.

**Recommended decision now:** approve the separately named HSP-S feature-based
Hit vs Non-Hit experiment. This published CC-BY dataset avoids the immediate MP3
requirement. File completeness, metadata joins, negative-label assumptions and
grouping still need checks before fitting. Our inspected chart columns show
3,870 charted / 3,866 without chart entries; no performance is promised.
The AcousticBrainz 263-ID subset is a closer-cohort alternative if retaining the
current song list matters more than using an existing published labelled benchmark.

No new Hugging Face subscription, token or Spotify plugin is needed to make that
scope decision. No additional model or accuracy testing was run on an unapproved
fallback. The original dataset, baseline and the 278-song metadata are preserved.

## Scientific limits

The selected positives are year-end songs; negatives are absent in a specified
public US weekly archive through the cutoff. This does not establish global or
permanent non-hit status. Same-artist/album selection and incomplete audio coverage
can cause sampling bias. Exact recording masters still need audio review. Public
chart evidence is not an authenticated Billboard export. Any future result is
retrospective classification, not a probability of future commercial success.
