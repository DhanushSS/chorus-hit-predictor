# Chorus Hit Predictor

### Audio-feature classification of Billboard chart association

An academic machine learning project with a reproducible training pipeline,
a verified trained model, and an interactive Streamlit application.

## Team

| Name | SRN |
| --- | --- |
| **Dhanush Sai Suprapadha** | **PES2UG24CS154** |
| **Deepthi V** | **PES2UG24CS150** |

## Project overview

The current project classifies songs using **436 whole-recording audio features**
from the published **HSP-S dataset**. An RBF Support Vector Machine distinguishes
records with Billboard chart entries from records without entries in that benchmark.

**Held-out accuracy: 73.07% · Balanced accuracy: 73.17%**

The project title is retained from our original chorus study. The current model
uses whole-recording Essentia features; it does not analyze 15-second choruses or
accept MP3 uploads. Missing chart entries are publisher labels, not independently
verified proof that a song never charted. Predictions describe this retrospective
benchmark, not future commercial success.

[Full results and graphs](docs/hit_nonhit/hsp_s_results/README.md) ·
[Method and reproduction](docs/hit_nonhit/hsp_s_protocol.md) ·
[Historical project archive](https://github.com/DhanushSS/chorus-hit-predictor-archive-2026-10-11)

## Results

| Metric | Nested development validation | Held-out test |
| --- | ---: | ---: |
| Accuracy | 70.97% | **73.07%** |
| Balanced accuracy | 70.94% | **73.17%** |
| Hit precision | 70.18% | 70.42% |
| Hit recall | 74.01% | 75.35% |
| Hit F1 | 72.04% | 72.80% |
| Macro F1 | 70.93% | 73.07% |
| ROC-AUC | Not recorded | 0.8076 |

The 95% connected-group bootstrap interval for test balanced accuracy is
**70.20–76.08%**. It reflects uncertainty within this dataset and fixed model,
not a guarantee for another music collection.

![Held-out confusion matrix](docs/hit_nonhit/hsp_s_results/confusion_matrix.png)

## Dataset and evaluation

- **7,736** published source records; **7,728** retained after duplicate/conflict checks.
- **3,865** chart-entry records and **3,863** records without chart entries.
- **6,235** development songs and **1,493** held-out test songs.
- Artist, recording, album and known duplicate links form connected identity groups;
  development and test groups are disjoint.
- **27** fixed configurations compared with **5 outer / 3 inner grouped folds**.
- Imputation, constant removal, scaling and optional PCA fit within training folds.
  Chart outcomes, listening counts and artist metadata are excluded from predictors.
- Selected model: **RBF SVM**, `C=1.0`, `gamma=scale`, balanced class weights.
- The locked test was accessed **once**, after development selection and bias checks.

Dataset/source bias remains a limitation. See the [diagnostics and completion
report](docs/hit_nonhit/hsp_s_results/handoff.md) for nuisance and permutation
results. Scores cannot be directly ranked against studies using different labels,
datasets or splits. Prediction scores are uncalibrated SVM margins, not probabilities.

## Run the application

Use **Python 3.12** and the pinned dependencies. No GPU or API token is required.

```bash
git clone https://github.com/DhanushSS/chorus-hit-predictor.git
cd chorus-hit-predictor
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/setup_model.py
python -m streamlit run app.py
```

Open `http://localhost:8501`, upload [`data/example_features.csv`](data/example_features.csv),
and select **Classify records**. The sample contains four development records and
is for interface demonstration only; its predictions are not new test evidence.

For your own input, use **1–100 feature rows**, at most **5 MB**, with the exact
436 columns and order provided by the downloadable template. Arbitrary song
metadata or MP3 files are incompatible with this model.

The [model bundle](models/README.md) is included in this repository. Setup verifies
its archive checksum, original file manifest, code identity and package versions
before extracting it into ignored local storage. It does not retrain or change the
reported results. Install dependencies online once; packaged-model prediction
then works offline. macOS and Linux are supported for the training workflow;
Windows training is unsupported.

## Reproduce the research experiment

The published model is ready for prediction after setup. To reproduce training,
use a fresh destination rather than overwriting the supplied completed run:

```bash
python -m chorus_hit.hsp_s download --out output/hsp_s/source
python -m chorus_hit.hsp_s prepare \
  --source output/hsp_s/source/hsp-s_acousticbrainz.parquet \
  --out output/hsp_s/reproduction_split
python -m chorus_hit.hsp_s train \
  --split output/hsp_s/reproduction_split --out results/hsp_s/reproduction_run
python -m chorus_hit.hsp_s evaluate \
  --split output/hsp_s/reproduction_split --run results/hsp_s/reproduction_run
```

The download verifies the full source checksum. Fixed group splits, per-fit time
limits and immutable manifests apply. Do not remove an access ledger or change
seeds to pursue a higher score. See the [protocol](docs/hit_nonhit/hsp_s_protocol.md)
for the exact rules and interrupted-run handling. Reproduction does not automatically
replace the app's supplied evaluated model.

## Verification

```bash
python -m pip check
python scripts/setup_model.py
python -m pytest -q
```

Tests cover feature/label contracts, connected groups, fold-local preprocessing,
one-use evaluation, supervised fit timeouts, model archive integrity, example
predictions and the current app. The former 285-test suite remains with the
archived project; it includes tests for retired datasets and interfaces.

[Latest automated checks](https://github.com/DhanushSS/chorus-hit-predictor/actions/workflows/tests.yml)

## Repository layout

| Folder or file | Purpose |
| --- | --- |
| `app.py` | Current Streamlit prediction application |
| `chorus_hit/hsp_s/` | Source audit, grouped training, evaluation and prediction |
| `chorus_hit/hit_nonhit/` | Shared training and validation dependencies |
| `configs/` | Fixed HSP-S model-search configuration |
| `models/` | Compressed evaluated model and checksum receipt |
| `data/` | Small licensed feature example for the demo |
| `scripts/` | Verified model installation and report export |
| `tests/` | Current pipeline and application regression tests |
| `docs/` | Results, protocol, figures and archive/migration record |

## Historical work

The previous chorus datasets, older models, completed experiments, candidate lists
and earlier branch histories are preserved in a separate public repository:

**[chorus-hit-predictor-archive-2026-10-11](https://github.com/DhanushSS/chorus-hit-predictor-archive-2026-10-11)**

The unfinished 278-song repeated-chorus experiment is archived with its metadata
and acquisition findings. It obtained no permitted exact recordings and produced
no new chorus-feature rows. The current HSP-S result does not resolve that audio
blocker. See the [migration record](docs/repository_migration.md) for what moved
and how the backup was verified.

## Attribution

HSP-S: Michael Vötter, Maximilian Mayerl, Günther Specht and Eva Zangerle (2021),
*Novel Datasets for Evaluating Song Popularity Prediction Tasks*,
[DOI 10.5281/zenodo.5383858](https://zenodo.org/records/5383858), **CC BY 4.0**.
Changes include scalar-feature selection, duplicate/conflict exclusions,
identity grouping, model fitting and evaluation. Labels reflect the publisher's
Billboard matching through 6 July 2019. See [NOTICE](NOTICE) and the
[example-data attribution](data/README.md).

No copyrighted recordings, credentials or private local paths are distributed.
