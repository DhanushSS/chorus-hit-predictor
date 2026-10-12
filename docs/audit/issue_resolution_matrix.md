# Issue resolution matrix

Audit baseline and branch base: `c166dddf3f0e639ba39783edf9b67eb959f6e065`.
The supplied 24-item plan was reconciled against that exact latest-main commit.
No baseline model/data evidence was replaced. Status as of 12 October 2026.

Commit keys: **R** = `4d028c9` registration + `fdc0e1f` logging correction;
**A** = `7a4bd60` tested app/audio changes; **D** = documentation/results delivery
commit, identified in [final delivery report](final_delivery_report.md).
`FIXED` means the applicable software/documentation acceptance passed; it does
not turn source-data limitations into solved scientific questions.

| Issue | Status | Implementation / evidence | Commit | Verification / remaining limit |
| --- | --- | --- | --- | --- |
| ISSUE-001 | FIXED | MP3 upload, playback, measurements; explicit preview-only fallback | A | Real generated MP3 in decoder, AppTest and browser; no hit prediction |
| ISSUE-002 | PARTIAL | Pinned isolated MusicExtractor, strict scalar mapper and parity comparator | A, D | macOS/Linux 381/436 coverage; 55 missing fields, zero same-master references; inference disabled |
| ISSUE-003 | FIXED | 25 MiB MP3 and 5 MiB CSV widget/parser caps; duration/rate/channel/decoded-size guards | A | Real corrupt/spoof/silent/truncated inputs; stream-boundary mocks; timeout and cleanup tests |
| ISSUE-004 | FIXED | UI/audio modules outside frozen implementation glob | A | Original ZIP/manifest/code hashes and four exact sample predictions preserved; fresh install |
| ISSUE-005 | FIXED | CSV template/sample, cell/schema errors, explicit reorder, preview and downloads | A | Valid CSV plus wrong order, duplicate, missing, extra, NaN/Inf, UTF-8, rows and size regressions |
| ISSUE-006 | FIXED | Seen-training-demo note in app and README | A, D | AppTest and browser show note; sample accuracy never used as validation |
| ISSUE-007 | PARTIAL | Registered 11-candidate nested development study; 226 completed fits | R, D | 71.69% accuracy / 71.65% BA; 80% not reached; no untouched external test |
| ISSUE-008 | PARTIAL | Codec/class/duration/sample-rate diagnostics and MP3-only grouped control | R, D | MP3-only BA 68.79%, with interval; source bias remains, no full matched-source cohort |
| ISSUE-009 | PARTIAL | Global nulls, class-pure/mixed groups, original within-group null interpretation | R, D | Global BA 50.17/47.58/49.17%; no p-value or zero-bias claim; era metadata absent |
| ISSUE-010 | PARTIAL | Graph disjointness, size/errors, no-largest-group sensitivity, bootstrap intervals | R, D | Zero exact duplicate development vectors; unknown aliases/masters/genre/era remain unverified |
| ISSUE-011 | FIXED | Publisher benchmark label scope throughout UI and README | A, D | Fixed literal classes; no lifetime-non-hit or future-success claims |
| ISSUE-012 | FIXED | Accurate matching app/README title and prominent whole-recording method | A, D | Browser and AppTest title; historical chorus origin linked separately |
| ISSUE-013 | FIXED | Dedicated development-only runner, no test evaluator invocation | R, A, D | Synthetic unreadable-test regression; original ledger and locked file hashes unchanged |
| ISSUE-014 | FIXED | Every completed-run fit logs warnings, failure state and time | R, D | 226 completed, zero failed/warned; preliminary interrupted trial records retained |
| ISSUE-015 | BLOCKED | MP3 accuracy explicitly unavailable; inference cannot be enabled by uploaded proof | A, D | Zero reference pairs / independent labelled MP3 cohort; numerical-pair pass cannot bypass gate |
| ISSUE-016 | FIXED | Five app tabs, metric cards, actual plots, prediction chart/download and audio preview | A | AppTest, actual browser uploads, desktop and narrow-layout inspection |
| ISSUE-017 | FIXED | Professional README with real CI badge, screenshots, workflow, results and attribution | D | Local relative-link/image tests; real captured screens; quick start checked in fresh clone |
| ISSUE-018 | FIXED | Historical handoff labelled; current operations guide added | A | Current app entry point exists; old pages instructions explicitly superseded |
| ISSUE-019 | FIXED | Legacy constants marked, import usages audited and current paths documented | A | No constant deletion or model hash change; app uses bundle schema and current run |
| ISSUE-020 | NOT REPRODUCED | Full verified-load benchmark published | A | Median 0.113 s load / 0.132 s four-row prediction; no excessive latency shown, no unsafe cache added |
| ISSUE-021 | FIXED | Expanded input, audio, model-integrity, research and AppTest regressions | A, D | Original 26 tests retained; expanded suite and clean CI evidence in final report |
| ISSUE-022 | FIXED | Separate pinned optional environment and generated extractor probe | A, D | Native macOS and Linux CI probe run; missing dependency fallback tested; Windows unsupported |
| ISSUE-023 | FIXED | Margin semantics, class direction, development feature-family sensitivity | A, R, D | No probability display; family drops labelled noncausal; no calibrator claimed |
| ISSUE-024 | DEFERRED | Actual local screenshots and fresh-clone demo delivered | D | Public hosting/walkthrough video optional and not deployed; no fake live demo or commercial audio |

**Totals:** 16 fixed, 5 partial, 1 blocked, 1 not reproduced, 1 deferred.

The exact scientific blockers are documented in
[feature compatibility](../audio_upload/feature_compatibility.md) and
[new research](../research_v2/model_improvement_report.md). No software task is
being left pending on a user answer. MP3 validation would require legally usable
same-master audio/reference pairs and later an independent labelled audio cohort.
