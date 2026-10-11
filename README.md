# Chorus Hit Predictor

An academic machine learning project exploring how audio features relate to
Billboard chart presence, with reproducible evaluation and a Streamlit demo.

**Team:** Dhanush Sai Suprapadha (PES2UG24CS154) · Deepthi V (PES2UG24CS150)

**Latest completed study:** HSP-S whole-recording features — **73.07% accuracy**
and **73.17% balanced accuracy** on a held-out test of 1,493 songs.

[Results and graphs](docs/hit_nonhit/hsp_s_results/README.md) ·
[Completion report](docs/hit_nonhit/hsp_s_results/handoff.md) ·
[Evaluation protocol](docs/hit_nonhit/hsp_s_protocol.md)

## Project overview

The repository contains three related pieces of work. Their datasets, feature
methods and results must be interpreted separately.

| Study | Data and method | Status |
| --- | --- | --- |
| **HSP-S feature study** | 7,728 accepted songs; 436 whole-recording Essentia descriptors; publisher chart-entry labels | Trained, tested once on a locked holdout, and available locally through feature-CSV prediction |
| **Original chorus project** | 751 songs; 518 supplied chorus features, with 74 variation features used by the active model; year-end hits vs other charted songs | Preserved model and original audio-upload demo |
| **278-song chorus extension** | 139 year-end hits and 139 non-charting candidates; planned 15-second chorus extraction | Metadata and pipeline ready; permitted exact recordings unavailable |

**The 73.17% result belongs to HSP-S.** It does not measure the accuracy of the
original MP3-upload demo or the unfinished 278-song chorus extension. HSP-S class 0
means no chart entry in the publisher's benchmark, rather than independently
verified lifetime non-charting status.

## Latest results

The selected model is an **RBF Support Vector Machine**, with `C=1.0`, balanced
class weights and `gamma=scale`. It uses all 436 eligible scalar audio features.

| Metric | Nested development validation | Locked test |
| --- | ---: | ---: |
| Accuracy | 70.97% | **73.07%** |
| Balanced accuracy | 70.94% | **73.17%** |
| Hit precision | 70.18% | 70.42% |
| Hit recall | 74.01% | 75.35% |
| Hit F1 | 72.04% | 72.80% |
| Macro F1 | 70.93% | 73.07% |
| ROC-AUC | Not recorded | 0.8076 |

The test balanced-accuracy **95% connected-group bootstrap interval is
70.20–76.08%**. This interval is conditional on the dataset and trained model;
it does not guarantee the same performance on another collection.

![HSP-S held-out confusion matrix](docs/hit_nonhit/hsp_s_results/confusion_matrix.png)

### How the evaluation works

- Full HSP-S source: **7,736 records**, verified against the publisher's checksum.
- After documented duplicate/conflict exclusions: **7,728 songs**, comprising
  **3,865 chart entries** and **3,863 records without chart entries**.
- Development partition: **6,235 songs** in **2,106 connected identity groups**.
- Locked test: **1,493 songs** in **527 groups**, accessed once after selection.
- **27 fixed model configurations**, compared with five outer and three inner
  grouped folds. All **486 inner validation fits** completed; warnings are retained.
- Artist, recording, album and known duplicate links are grouped before splitting.
- Imputation, constant-feature removal, scaling and optional PCA fit only on each
  training fold. Chart outcomes, artist names and listening counts are excluded
  from model predictors.

The source/bias checks and their limitations are documented in the
[completion report](docs/hit_nonhit/hsp_s_results/handoff.md). Passing those checks
does not prove an absence of dataset bias. These are retrospective classification
results, not calibrated probabilities of future commercial success. Different
labels, datasets and evaluation procedures prevent a direct ranking against
Eric Liu's reported results or the original chorus model.

## Quick start

Use **Python 3.12** with the pinned dependencies. These commands select the feature
branch containing the HSP-S implementation; it has not been merged into `main`.

```bash
git clone --branch feat/hit-vs-nonhit-v1 https://github.com/DhanushSS/chorus-hit-predictor.git
cd chorus-hit-predictor
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
python -m streamlit run app.py
```

Open the address printed by Streamlit, normally `http://localhost:8501`.

| App page | Input | Availability |
| --- | --- | --- |
| Main page — Chorus Hit Predictor | Stored-song examples or permitted audio uploads | Uses the preserved original chorus model; uploaded-audio accuracy is unverified |
| Hit vs Non-Hit feature study | CSV with 1–100 matching HSP-S feature rows in the exact column order | Requires a locally reproduced, evaluated HSP-S model |

**The trained HSP-S bundle is not checked into GitHub.** A fresh clone can open the
application, but must complete the following reproduction steps before the HSP-S
page can classify records. The template on that page specifies the feature columns.
It cannot classify MP3s using the original librosa chorus extractor.

macOS is locally tested; Linux runs the automated checks. Windows model training
is unsupported by the current supervision implementation. Installation and the
HSP-S source download require internet access; no GPU, Spotify connection or
Hugging Face token is required for this workflow.

## Reproduce the HSP-S experiment

From the repository root, with the virtual environment active:

```bash
# Download the complete published feature file and verify its checksum.
python -m chorus_hit.hsp_s download --out output/hsp_s/source

# Audit records and freeze the group-separated development/test partition.
python -m chorus_hit.hsp_s prepare \
  --source output/hsp_s/source/hsp-s_acousticbrainz.parquet \
  --out output/hsp_s/split_001

# Select and train using development data only.
python -m chorus_hit.hsp_s train \
  --split output/hsp_s/split_001 --out results/hsp_s/run_001

# Evaluate once, only if the predefined development checks pass.
python -m chorus_hit.hsp_s evaluate \
  --split output/hsp_s/split_001 --run results/hsp_s/run_001
```

These are first-run commands. Completed runs cannot be overwritten. For an
interrupted, unfinished training run, use `train --resume` with the same paths
and unchanged data, code, configuration and environment. Do not remove the
locked-test access ledger to evaluate again, or change seeds to pursue a score.

Source data and split files remain in ignored `output/hsp_s/`; the model and
trial artifacts remain in ignored `results/hsp_s/`. GitHub contains the code,
configuration, tests and safe aggregate reports. See the
[full protocol](docs/hit_nonhit/hsp_s_protocol.md) for the source, grouping rules,
fit budgets and failure handling, and the
[handoff](docs/hit_nonhit/hsp_s_results/handoff.md) for report-export instructions.

## Tests and verification

```bash
python -m pytest -q
python -m pip check
python -m chorus_hit.readiness
```

The HSP-S implementation at commit `4910054` passed **285 tests** locally and in
[GitHub CI](https://github.com/DhanushSS/chorus-hit-predictor/actions/runs/38046487532).
Four existing warnings were reported. CI checks software and synthetic pipeline
fixtures; it does not reproduce the full HSP-S research experiment. A subsequent
local browser check uploaded four real development feature records and verified
that the prediction table matched the model's outputs. That demo is not an
additional accuracy test.

All **54 protected original artifacts** matched their recorded hashes at handoff.
The later report and README commits do not change the trained model or its metrics.

## Original chorus work and remaining audio blocker

The preserved original model achieved **54.61% accuracy / 54.66% balanced accuracy**
in artist-grouped nested development evaluation. Its target is year-end hits vs
other charted songs. No untouched test remains for that historical experiment;
its earlier exploratory scores must not replace the nested assessment.

The 278-song extension retains its candidate metadata and 15-second chorus/518-feature
pipeline. **Zero permitted exact recordings and zero new chorus-feature records
were obtained.** Completing that experiment requires lawful access to the matching
recordings for local analysis, or compatible licensed chorus features with recording
identities and extraction provenance. The HSP-S fallback does not remove this blocker.

- [278-song completion status](docs/hit_nonhit/final_48h_status_278.md)
- [Audio acquisition findings](docs/hit_nonhit/audio_feasibility_278.md)
- [Academic workflow](docs/hit_nonhit/academic_workflow.md)
- [Original audio contract](docs/audio_contract.md)
- [Historical evaluation registry](docs/evaluation_registry.json)
- [Protected artifact manifest](docs/hit_nonhit/preservation_manifest.json)

## Repository structure

| Location | Purpose |
| --- | --- |
| `app.py` | Original Streamlit application |
| `pages/` | Separate HSP-S feature-study page |
| `chorus_hit/hsp_s/` | HSP-S download, preparation, training, evaluation and prediction |
| `chorus_hit/hit_nonhit/` | Hit/non-hit audio and metadata infrastructure |
| `chorus_hit/` | Preserved chorus feature and model pipeline |
| `configs/` | Feature contracts and experiment configurations |
| `data/` | Original feature dataset, provenance and candidate metadata |
| `tests/` | Integrity, leakage-prevention and application regression tests |
| `scripts/report_hsp_s.py` | Export saved HSP-S aggregates and figures |
| `docs/hit_nonhit/hsp_s_results/` | Published results, figures and completion report |

## Data sources and attribution

**HSP-S:** Michael Vötter, Maximilian Mayerl, Günther Specht and Eva Zangerle (2021),
*Novel Datasets for Evaluating Song Popularity Prediction Tasks*,
[Zenodo DOI 10.5281/zenodo.5383858](https://zenodo.org/records/5383858), **CC BY 4.0**.
Our changes include scalar feature selection, duplicate/conflict exclusions,
connected-group partitions and nested model evaluation. Labels reflect the
publisher's matching of Billboard records through 6 July 2019.

**Original chorus features:** adapted from
[Predicting Hit Songs Using Repeated Chorus](https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus)
under Apache-2.0; see [NOTICE](NOTICE), [licenses](licenses/) and
[data provenance](data/provenance.json).

The repository remains public. Original datasets, completed results and the active
chorus model are preserved. Copyrighted recordings, credentials and private local
paths are excluded from the public experiment outputs.
