# Delivery report — 12 October 2026

## Scope and branch

- Base: latest `main` at `c166dddf3f0e639ba39783edf9b67eb959f6e065`, matching audit HEAD.
- Feature branch: `feat/mp3-csv-ui-readme-research-v2`.
- Main and the public historical archive remain untouched; no merge authorized or performed.
- All 24 issue statuses: [resolution matrix](issue_resolution_matrix.md).

## Delivered

Working strict CSV inference, bounded local MP3 preview, five Streamlit tabs,
real model charts/screenshots, clearer score/label semantics, isolated optional
Essentia experiment, new regression tests and a professional README with both
team members/SRNs. Frozen model code was not modified. Historical config constants
remain for shared imports and are explicitly marked.

Actual browser checks uploaded the licensed four-row feature example, produced
all four original classes/margins, and uploaded a software-generated 15-second
MP3. The latter showed audio controls, 44,100 Hz, one channel, duration, envelope
and an explicit no-prediction message. Desktop and narrow layout were inspected.
[Screenshots](../assets/app_overview.jpg), [CSV](../assets/app_csv.jpg),
[MP3 preview](../assets/app_mp3_preview.jpg) are actual captures.

## Results and limitations

| Evidence | Outcome |
| --- | --- |
| Original test | 73.07% accuracy / 73.17% BA, unchanged |
| New nested development | 71.69% accuracy / 71.65% BA; 95% BA interval 69.91–73.59% |
| Same-fold fixed SVM baseline | 71.64% accuracy / 71.60% BA |
| New final development selection | Hard voting; selection CV BA 71.64%, not independent test |
| New fits | 226 completed, 0 failed, 0 warned; 571.3 s accumulated trial time |
| New untouched test | NOT AVAILABLE |
| Target decision | 80% not reached; no new model promotion |
| Optional extractor | Real macOS/Linux generated-MP3 extraction, 381/436 exact fields |
| MP3 classification | Disabled: missing schema fields and no same-master parity/reference cohort |

The new nested procedure differs from the original 27-candidate selection study;
its small change is not proof of a reliable improvement. Compared with the fixed
SVM in the same folds, the gain is only about 0.05 percentage points. Bias,
metadata gaps and repeated-research uncertainty remain. All configurations,
controls, confusion matrix and hashes are in the
[research report](../research_v2/model_improvement_report.md).

## Baseline preservation

Model ZIP SHA-256:
`4c335301d937ecacbdc53445d435443f5aa19badbb84ad6bb386c51a81b9f7d3`.
Original access ledger remains USED_ONCE. Protected file hashes, including the
locked Parquet, were compared before and after; no locked labels were deserialized
for new research. Hash receipt: [baseline_preservation.json](baseline_preservation.json).
Original four scores: `1.187263773345118, -0.27028194326293875,
1.066465614339061, 0.5116200719057967`. No model, ledger, historical result or archive
was rewritten. A separate package keeps frontend changes outside frozen code hashes.

## Commands and checks actually executed

- Fresh baseline clone: `python scripts/setup_model.py`, `python -m pytest -q`
  (**26 passed**), `python -m pip check` (clean). Used the existing pinned environment;
  this local baseline check was a fresh source clone, not a fresh dependency install.
- Current checkout: `python scripts/setup_model.py`; `python -m pytest -q`
  (**71 passed** before final documentation tests); `python -m pip check`; `git diff --check`.
- `python -m chorus_hit.research_v2.run --split output/hsp_s/split_001 --out output/research_v2/run_002`.
- `python scripts/summarize_research_v2.py --split output/hsp_s/split_001 --run output/research_v2/run_002 --out docs/research_v2`.
- Separate Python 3.12 Essentia environment: `pip install -r requirements-audio-probe.txt`;
  `python scripts/probe_essentia.py --out docs/audio_upload/macos_probe.json`.
- Real Streamlit session: `python -m streamlit run app.py`; browser CSV/MP3 upload checks.
- [CI at app commit 7a4bd60](https://github.com/DhanushSS/chorus-hit-predictor/actions/runs/38162493625):
  **success**; fresh pinned dependency installs, pip check, model setup and tests
  on Ubuntu 24.04 and macOS 15; separate Linux Essentia probe successful.
  [Linux probe receipt](../audio_upload/linux_probe.json).

Final delivery commit/PR and final-SHA CI are recorded in the publication section
once verified. Local generated audio, raw source data and private paths are excluded.

## Commit sequence

- [4d028c9](https://github.com/DhanushSS/chorus-hit-predictor/commit/4d028c9): registered bounded research and supplied audit.
- [fdc0e1f](https://github.com/DhanushSS/chorus-hit-predictor/commit/fdc0e1f): record control-fit warnings/time; preserve interrupted attempt.
- [7a4bd60](https://github.com/DhanushSS/chorus-hit-predictor/commit/7a4bd60): tested CSV/MP3 interface, optional extractor, safeguards and CI.
- Documentation/results delivery and publication receipt: see publication section below.

## Remaining input

To continue MP3 classification validation, supply legally analyzable **identical
recording masters plus their original 436-feature vectors and extraction-profile
provenance**. This is not required to use the completed CSV app or MP3 preview.
A genuinely untouched compatible labelled evaluation cohort is also required
before any new independent accuracy claim. No public deployment has been created.
