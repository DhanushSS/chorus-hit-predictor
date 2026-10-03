# Runtime audit — October 1–2, 2026

This audit executes the repository locally. The supplied external brief inspected source/artifacts but did not execute them; its limitations remain distinct.

## Phase A evidence

- Audited checkout and tree match `0e9e4628d0a72af79694bc555d64a562e83dcfa2` / `25db599399d402490b220a0e1140f5e348193edf`; initial tree clean.
- Complete original archive: `../Chorus_Hit_Baseline_V1.zip`. Every original tracked file hash: `results/baseline/v1_manifest.json`. Original data/model/results/docs remain intact.
- Existing isolated Python 3.12 environment: exact requirements installation and `pip check` succeed. Original six tests pass. Commands, durations, exits and logs: `results/audit/phase_a_commands.json`.
- Disposable full V1 training: exit 0, 49.76 seconds including startup. All nine model metrics match to 1e-12; selected model and all 154 predictions match. See `v1_reproduction_command.json`, `v1_reproduction_comparison.json` and `v1_reproduction.log` in `results/audit/`.
- Independently recomputed: 751 rows, 518 raw columns, 71 normalized artist strings, labels 385/366; development 597, historical test 154. V1 PCA retains 162 dimensions. No exact feature duplicates or normalized artist/title duplicates. No original audio in the authorized project directory.
- Prepared SHA-256 matches `79951ffc393fc836405509326d1ee03efd96434327d800711c4b8ed817ef44ce`. Saved model package versions, feature order, labels, score orientation and predictions verified.
- Integrity tests pass before alternate experiments. The exact legacy import is verified by source byte hash. Alternate import tests reject false provenance and preserve IDs across row reorderings.

## Finding status

| Finding | Runtime status / action |
|---|---|
| F01 target / label audit | Confirmed year-end vs other-chart target; no changed labels. Hash-selected six-record sample; official pages returned HTTP 402, so primary verification remains incomplete. Full audit manifest added in Phase C. |
| F02 bottleneck | Hypothesis unresolved; Phase B records train/validation gaps, dimensions, groups and learning curves. |
| F03 limited representation search | Confirmed; bounded runner adds controlled comparisons. |
| F04 historical test | Confirmed; frozen original membership excluded from new tuning. |
| F05 identity strings | Confirmed limitation; current comparable run retains artist-string groups. Canonical identity evidence is unresolved; strict performer claims are not made. |
| F06 wrong custom-path hash | Reproduced in original source; resolved with actual path/byte identity and separate frame fingerprint; regression tested. |
| F07 false import attribution / IDs | Resolved: exact source hash or explicit alternate provenance; stable alternate IDs; no overwrites. |
| F08 waveform parity | Still unverified; versioned legacy extractor preserved; new recording interfaces in Phase C. |
| F09 mixed artifacts/cache | Resolved by shared hash/schema/run/label loader and immutable run identity; app/CLI/report integration tested in Phase E. |
| F10 stale report text | Resolved: common summary/report content, dynamic conclusion/schema tests, executed notebook and reviewed V2 exports. |
| F11 shared output writes | Resolved: immutable V2 runs, checkpointed incomplete runs, explicit baseline wrapper; V1 command requires a new output directory. |
| F12 coverage | Existing six preserved; additional provenance, input compatibility, target, schema and runner tests added; 20 local tests passed; CI configured but not pushed or run remotely. |
| F13 export vs usefulness | Explicit evidence status and no automatic promotion. Dummy remains eligible in V2. |
| F14 missing audio / forecasting | Confirmed missing recordings, rights/evidence/time manifest and fresh evaluation collection; interface work can proceed. |

## Evaluation protocol locked before Phase B

Both JSON configurations contain the same 35 candidate settings before either is executed. Quick: 3 grouped folds. Thorough: 5 outer grouped folds, each with 3 inner grouped folds selecting across all candidate families, followed by 3-fold selection on the full development partition. Seeds 42/43/44 are fixed. Default estimator thresholds; no calibration, ensembling or adaptive seed search. The logistic family has 13 bounded settings (including its V1 control); total candidate budget 35. Two CPU workers; single-threaded fits; quick budget 1,200 seconds, nested budget 3,600 seconds; per-wait fit limit 180 seconds. Resume retains completed folds and requires unchanged code/config/data identity.

Tuning maxima, nested estimates, historical results and fresh tests have distinct labels. The nested procedure's pooled AUC is omitted because different outer models may have incomparable score scales; per-fold AUC is retained. Whole-artist bootstrap intervals condition on recorded predictions, and do not include all selection/training uncertainty. Fold standard deviation is not a confidence interval.
