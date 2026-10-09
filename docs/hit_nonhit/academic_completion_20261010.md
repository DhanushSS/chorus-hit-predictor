# Academic revision completion report — 10 October 2026

## Direct answer

**PARTIAL: the new target does not yet have a trained, evaluated real-audio classifier.**
The implemented pipeline and application are tested. The project now has real,
source-linked metadata candidates and a balanced acquisition list. It still has
**zero permitted real recordings and zero genuine new feature rows**. Therefore
all new-target development/test metrics are **NOT AVAILABLE**. No accuracy was invented.

## Repository and target

- Repository remains public: `DhanushSS/chorus-hit-predictor`.
- Separate branch: `feat/hit-vs-nonhit-v1`.
- Existing draft PR: [#6](https://github.com/DhanushSS/chorus-hit-predictor/pull/6), **not merged**.
- Main verified at `853e1b88fbc6a10198bda9eff1529766f2201d83` before publication.
- Starting feature-branch HEAD: `b4f64e79ff6fe8fada665cd6e0e25d161d3f34f8`.
  The final implementation HEAD is the commit containing this report; see the PR.
- Active old model remains `v4_std74_001`, using the original standard-deviation
  features and logistic regression. Its saved artifacts and deployment choice are unchanged.
- New academic cohort: reused **2006–2021 year-end US Hot 100 hits** versus comparable
  songs absent in the audited public weekly archive from first release through
  **30 December 2023**, with at least 24 months' follow-up.
- Metadata source status: **available and structurally audited**, with explicit
  public-source/identity limitations. Legal local audio status: **unavailable**.
- Faculty approval for the revised target: **pending submission checkpoint**.
  The earlier approval concerned the old adaptation; it is not silently transferred.
  Development proceeds without paperwork or a commercial chart subscription.

## Dataset evidence

| Item | Actual count/status |
| --- | --- |
| Original rows/features preserved | 751 songs, 518 features |
| Original positive candidates reused for audit | 366 |
| Initial exact-name annual + weekly verification | 344; 22 pending alias/title review |
| Original weekly-only class excluded from negatives | All 385 |
| Weekly issues audited, 2000–2023 | 1,253 expected / 1,253 structurally complete; no missing weeks |
| Annual metadata, 2006–2021 | 1,600 ranked entries |
| Discovered recording candidates | 499 |
| Year-end positives with metadata evidence | 182 |
| Non-charting candidates under the public archive rule | 263; three lack a comparable positive |
| Unknown/excluded for identity or version review | 54: 38 unresolved metadata, 11 release/chart conflicts, five alternate mixes |
| Balanced acquisition list | **278 songs: 139 positives + 139 negatives** |
| Lead artists / canonical artists including guests | 36 / 52 |
| Albums / match sets / distinct musical works | 67 / 83 / 278 |
| Connected metadata groups | 34 |
| Metadata-only split support | 27 development groups / 7 held-out groups; nested five-by-three folds feasible |
| Permitted real audio / genuine feature rows | **0 / 0** |
| Real reviewed chorus excerpts | **0** |

All selected songs have same-album matches. Each original positive and musical work
is acquired at most once. The final metadata preflight contains 228 development
songs (114 per class) and 50 held-out songs (25 per class). **This is not a frozen
or evaluated final test.** Available audio and acoustic duplicate groups must be
checked before freezing actual feature data. The seed was not searched or changed.

The discovery retained two explicit errors: ambiguous `Ke$ha` artist search, and
no usable dated official studio edition for the `Days Before Astroworld` candidate.
Other artists and albums continued. Unknown candidates were not guessed into either class.

“Verified non-charting” means absent from this audited public archive under the
stated artist/title/version rules. It is not Billboard certification, proof of
never charting anywhere, or a future prediction. Full source snapshots remain
local; the public repository contains derived metadata, source URLs and hashes.
No recordings, credentials, commercial archive exports, new PDFs or slides were committed.

## Phase-by-phase status

| Phase | Status | Changed/used paths | Tests and evidence | Remaining blocker |
| --- | --- | --- | --- | --- |
| 0 — Preserve and audit | DONE | `preservation_manifest.json`, existing active model/data; latest baseline audit | All 54 protected file hashes match; old loader and CH0001 demonstration still work | None |
| 1 — Target and feasibility | PARTIAL | `configs/hit_nonhit_academic.json`, `public_sources.py`, `academic_workflow.md` | Pinned public GitHub commits and Wikipedia revision; cached MusicBrainz metadata; 1,253 complete weeks | Permitted local audio; faculty checkpoint before submission |
| 2 — Chart/identity pipeline | DONE for academic metadata pilot | `academic.py`, `academic_evidence.py`, source/classification adapters; `data/hit_nonhit_v1/academic/` | Actual 499-candidate audit, 278-song balanced list; hidden hits, aliases, coverage gaps, related versions and conflicts tested | 54 unresolved candidates excluded; public metadata authority limitations remain |
| 3 — Shared audio features | PARTIAL | `intake.py`, `build_dataset.py`, existing shared audio extractor | Generated test audio passes both-class metadata-to-feature integration; exactly 15 seconds / 518 features; intake reports 278 missing recordings | Real permitted recordings and human chorus review |
| 4 — Group audit and split | PARTIAL | `academic.metadata_preflight`, existing `freeze_split.py` and `audit.py` | Metadata supports 34 groups and nested folds; unique works in acquisition list | Acoustic duplicate checks and genuine features before freezing |
| 5 — Training/evaluation | PARTIAL | Academic config, `train.py`, existing bounded modeling/evaluation modules | 27 configured candidates; all518/std74/PCA; small-model skip rules tested; synthetic end-to-end smoke | Genuine dataset; no training/test metrics yet |
| 6 — Application and inference | PARTIAL | `ui.py`, `artifacts.py`, existing prediction CLI | Headless app checks; intake download and actual metadata progress; cohort-specific prediction names | New inference requires a genuinely trained and evaluated bundle |
| 7 — Tests and CI | Local checks executed; see verification section | `tests/hit_nonhit/test_academic.py`, existing full suite and CI | Public-source and intake regression tests plus all existing guards | Remote check outcome is shown on the PR |
| 8 — Documentation and PR | DONE locally; published through draft PR | README, academic workflow/report, historical-status notices, metadata summaries | No misleading accuracy claim; main unchanged; preserved old results | Merge remains a separate user decision |

## Modeling and performance

The fixed academic configuration compares 27 candidates covering dummy, logistic
regression, linear/RBF SVM, random forest, extra trees, histogram gradient boosting
and a small regularized neural network. Both 518 and 74 standard-deviation feature
representations are included; two logistic/PCA variants use fold-local PCA.
MLP/boosting are skipped, without blocking simpler models, if there are fewer than
15 development groups. Thirty seconds per fit and an 18,000-second declared upper
bound prevent unbounded tuning.

All feature selection/scaling/compression remains within training folds. The
connected grouping, one-use final-test ledger, source/nuisance and permutation
checks remain in force. No threshold or split seed was selected from results.

| Metric | New target |
| --- | --- |
| Nested development balanced accuracy / accuracy | NOT AVAILABLE |
| Locked test balanced accuracy / accuracy | NOT AVAILABLE |
| Precision, recall, F1, ROC-AUC, average precision | NOT AVAILABLE |
| Confusion matrix / bootstrap interval | NOT AVAILABLE |
| Real inference speed / chorus detector accuracy | NOT AVAILABLE |

The old **54.61% accuracy** remains historical evidence for year-end hits versus
other already-charted songs. It is not a new-target baseline or a directly
comparable improvement. A 70–75% balanced-accuracy aspiration is not an acceptance
shortcut and has not been reached or claimed.

## Exact checks and results

The final command outputs are preserved under `output/hit_nonhit/` locally.
CI uses synthetic fixtures and public checked-in metadata, without audio downloads.

| Exact command/check | Actual outcome |
| --- | --- |
| `.venv/bin/python -m pytest -q` | **253 passed**, four existing audio-library deprecation warnings, 161.67 seconds |
| `.venv/bin/python -m pytest tests/hit_nonhit/test_academic.py tests/hit_nonhit/test_audio_and_ui.py -q` | **49 passed**, 8.63 seconds, after the final optional-progress UI error-handling fix; includes 35 academic regression cases |
| `.venv/bin/python -m chorus_hit.hit_nonhit.smoke --out output/hit_nonhit/academic_smoke_release_20261010` | Passed; synthetic software test only, research accuracy null |
| `.venv/bin/python -m pip check` | No broken requirements |
| `.venv/bin/python -m chorus_hit.readiness` | Software ready; active old artifact valid |
| `.venv/bin/python -m chorus_hit.artifacts` | Active saved artifact loads |
| `.venv/bin/python -m chorus_hit.predict --run-id v4_std74_001 --track-id CH0001` | Existing training-row demonstration works; not a fresh test |
| Public `prepare` and `discover`/`curate` commands | Executed against real cached public metadata; final candidates and provenance published |
| `intake.ingest(..., dry_run=True)` on the supplied list | 278 intake rows, zero ready audio rows, 278 missing file paths; zero legacy features reused |
| SHA-256 comparison against preservation manifest | **54/54 unchanged**, zero mismatches |
| `git diff --check` | Passed |

The local full run preceded the last optional-progress UI exception guard; the
focused rerun includes both normal and corrupt-progress app paths. The draft PR
runs the full final checkout again in GitHub Actions. Its check result is shown
on the PR, rather than treating an earlier commit's CI as validation of this one.

Local logs: `academic_pytest_release.log`, `academic_final_focused.log`,
`academic_smoke_release.log`, `academic_readiness_release.json`, and
`academic_intake_dry_run.json`, under `output/hit_nonhit/`.


## Minimum next inputs

1. Supply local recordings for songs in `data/hit_nonhit_v1/audio_intake.csv`,
   with permission to process them. Both classes must use the same extraction pipeline.
2. Fill file paths, permission basis/reference, reviewer, recording confirmation,
   and preferably every 15-second chorus start. The intake command reports any
   missing/invalid input before extraction. No formal permission letter is required.
3. Confirm the revised target with faculty before submission. This is independent
   of development and is not the reason training is currently blocked.

For this checkout, the prepared archive is already at
`output/hit_nonhit/academic_prepared_final` and candidates are available at
`data/hit_nonhit_v1/academic`. Follow the [workflow](academic_workflow.md) to ingest,
extract, freeze, train, and evaluate once. No MusicBrainz rediscovery is required
just to use the supplied list.
