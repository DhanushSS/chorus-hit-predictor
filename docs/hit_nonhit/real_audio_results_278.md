# Real-audio results — 278-song cohort

**NOT AVAILABLE — NO VALID REAL-AUDIO TRAINING**

Status on 10 October 2026: `BLOCKED_AUDIO`. The source investigation is complete
within its cap; the real training experiment has not run.

| Measurement | Verified result |
| --- | --- |
| Selected metadata | 278: 139 year-end hits / 139 non-charting candidates |
| Permitted recordings obtained / processed | 0 / 0 |
| Real extracted 518-feature rows | 0 (0 per class) |
| Human-reviewed choruses | 0 per class |
| Actual audio-connected groups | 0; metadata-only preflight has 34 |
| Real dataset / split fingerprints | NOT AVAILABLE; no real package or split created |
| Models evaluated | 0 of the 27 preregistered configurations |
| Models skipped | All 27 blocked before fitting by missing real inputs |
| Best model / selected configuration | NOT AVAILABLE |
| Training / nested-validation metrics | NOT AVAILABLE |
| Locked-test access | **NOT_USED** for real data |
| Real locked-test size / class support | NOT AVAILABLE; no split frozen |
| Accuracy / balanced accuracy | NOT AVAILABLE |
| Precision / recall / F1 | NOT AVAILABLE |
| ROC-AUC / average precision | NOT AVAILABLE |
| Confusion matrix (TN, FP, FN, TP) | NOT AVAILABLE |
| 95% group-bootstrap interval | NOT AVAILABLE |
| Real extraction / training timing | NOT AVAILABLE |
| Real source/codec nuisance diagnostics | NOT RUN — NO AUDIO |

No model-comparison plot or confusion matrix is generated: there are no real
predictions to plot. Synthetic smoke training is kept in a separately marked,
ignored directory and reports `research_accuracy: null`. Its synthetic holdout
ledger is not a real experiment's locked test.

## Preserved protocol and fingerprints

- Intake SHA-256: `c5414777f3aca769e493f8eaac130052bd05d7c3b2eca429ae48791f6d4e4a98`.
- Academic config SHA-256: `35ebf9effd354c081920c85b91d525ae932f3233e40761d779ddff89e7d90ea0`.
- Extractor contract digest: `472afe19346e3956c9a82f4d174da2db66066166f23a9a311fe80af5aec192e7`.
- Ordered feature-schema digest: `276515c658485f263e336b9a31dfe4c86969ade87ccb28a0de4a222e2c1261a9`.
- Same mono 22,050-Hz / 15-second / 518-feature contract; no legacy feature splicing.
- Same fixed seed, group connectivity, fold-local transformations, nested validation,
  conditional model guards, fit limits and one-use final evaluation.
- Original 751-row dataset, saved models/results, active configuration and all 54
  protected hashes unchanged. No default deployment replacement.

## Interpretation limits

Class 0 means absent from the audited public US Hot 100 archive through 30 December
2023, not “never successful anywhere.” Candidate selection favors known hit artists
and their album tracks; this is a retrospective convenience sample. Public metadata
is not an authenticated Billboard export and cannot resolve every audio master.
AcousticBrainz availability for 263 IDs is a separate whole-recording feature finding,
not a new score or a completed chorus dataset. Historical scores on year-end versus
other charted songs must not be described as results for this target.

See [feasibility and fallback recommendation](audio_feasibility_278.md) and
[completion status](final_48h_status_278.md).
