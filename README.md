# Predicting Hit Songs Using Repeated Chorus

**UE24CS352A - Machine Learning mini-project**

| Student | USN |
| --- | --- |
| Dhanush Sai Suprapadha | PES2UG24CS154 |
| Deepthi V | PES2UG24CS150 |

A reproducible study of whether 15-second chorus audio features distinguish stronger chart success, with a working Streamlit demo, model comparisons, an executed notebook, reports, and presentation.

## Local evaluation audit (October 3)

[Corrected accuracy report and remaining work](docs/ACCURACY_REPORT_RECONCILIATION.md). The assigned target is unchanged. A fixed-candidate stability check across ten grouped partitions averaged **53.05% balanced accuracy** (range **49.03%–54.26%**); this is descriptive reuse of the development data, not a new independent score. All 100 fits completed and **26 tests passed**. V3/demo results remain unchanged; this audit is uploaded on the review branch linked below.

## Local Hugging Face assisted pilot (October 3)

[Measured findings and limitations](docs/HUGGING_FACE_PILOT_RESULTS.md): on 340 candidate-linked development songs, balanced accuracy was 50.79% for the matched legacy baseline, 52.09% for published musicnn features and 54.17% for combined inputs. The combined paired gain interval (-0.22 to +7.21 percentage points) includes zero. Recording/chorus equivalence is unverified. This subset pilot does not replace the V3 result or the active demo, and it is now included in the review branch.

## Latest: V3 robustness study (completed October 3)

[Read the V3 findings](docs/V3_RESEARCH_RESULTS.md) and [data-source review](docs/V3_DATA_SOURCE_REVIEW.md).

- Expanded to **55 model settings**, completing **1,059 recorded fits** with no failed trials.
- Nested balanced accuracy: **53.25%**, compared with V2 **52.85%**. The paired change is **+0.40 percentage points**, with an approximate 95% interval of **−2.71 to +3.04 points**. This is an inconclusive change.
- The final selected classifier has the same configuration and inspected learned parameters as V2. The expanded selection procedure has not established a better final model.
- This is exploratory reuse of previously inspected development data. No fresh test exists, and the strict >75% target remains unmet.
- **24 tests passed**. The V3 research option was verified in the running demo; original results and exports passed preservation checks.
- Uploaded branch: [`research/chorus-v3`](https://github.com/DhanushSS/chorus-hit-predictor/tree/research/chorus-v3). GitHub access was restored on October 3. This update is available for review and has not been merged into `main`.
- Matching song recordings are still missing. The reviewed sources do not supply the matched audio needed for the planned richer-representation study.

## October 2 audit and research update

The original run and submission exports remain preserved. **Start with [the V2 handoff](docs/AUDIT_HANDOFF.md)** and [reproduction commands](docs/REPRODUCE_V2.md). Updated report/presentation: `docs/v2/`; executed walkthrough: `notebooks/v2/Project_Walkthrough.ipynb`.

- All original scores and predictions reproduced in a disposable copy.
- Actual-file provenance, stable alternate imports, compatible artifact loading, immutable runs and resumable grouped searches implemented.
- 35 predeclared settings evaluated using the original 597-song development partition.
- Nested procedure balanced accuracy **52.85%** (approximate 95% artist-bootstrap interval **48.26%-57.34%**).
- Frozen V2 candidate (`lr_mi_50`: logistic regression with 50 MI-selected features) historical balanced accuracy **55.16%**, versus V1 **46.63%**, on the same 154 songs. Paired change interval **-1.59 to +16.86 percentage points** includes zero.
- Accuracy, balanced accuracy, positive-class precision, recall and F1 all remain below the strict **>75%** target. No fresh test exists. Historical improvement is not reliable confirmation.
- Original demo remains active (`configs/active_run.json`). The sidebar offers the V2 candidate as a research option.
- Audio/embedding interfaces have software tests. Real embedding experiments remain blocked by missing recordings, rights and verified identities/labels. See [required inputs](docs/AUDIO_DATA_REQUIREMENTS.md).
- Review branch: [`audit/chorus-v2`](https://github.com/DhanushSS/chorus-hit-predictor/tree/audit/chorus-v2). The update is uploaded and is not merged into `main`. [GitHub Actions passed](https://github.com/DhanushSS/chorus-hit-predictor/actions/runs/36970412334) for publication commit `2dfe0ee`. The earlier authentication blocker was resolved on October 3; the V3 branch includes this V2 work. See [GitHub checks](https://github.com/DhanushSS/chorus-hit-predictor/actions) for the latest status.

The numerical section titled **Original V1 results** below describes the preserved baseline only.

## What this experiment predicts

The project follows the idea in [Eric Liu's 2021 CS229 report](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf). The available licensed dataset has a different operational target:

- **1: Year-end hit** in the upstream Billboard year-end collection.
- **0: Other chart song** from sampled weekly Billboard Hot 100 charts, excluding titles already collected as year-end hits.

Both classes can have charted. This is a **year-end-hit proxy**, not an exact reproduction of the paper's charted/never-charted experiment. Labels are inherited from the source collection notebook and have not been independently checked against every chart. This distinction appears in the app, report, and presentation.

## Run the demo

On the prepared Mac, double-click **Run Demo.command**, or open a terminal in this folder and run:

```bash
.venv/bin/python -m streamlit run app.py
```

For a fresh machine, use **Python 3.12**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, use `py -3.12 -m venv .venv`, activate `.venv\Scripts\activate`, and run the same pip and Streamlit commands. `run_demo.bat` automates these steps. The demo opens at `http://localhost:8501`.

The dataset and trained model are included, so the basic demo works offline after installing dependencies. The audio formats WAV and FLAC work through SoundFile. MP3 decoding depends on codec availability; convert a problematic file to WAV with an installed audio converter.

### Demo modes

1. **Held-out song:** choose a song from the test partition and compare its predicted class with the dataset label. The artist and song were excluded from model training.
2. **Upload audio:** upload a 15-second chorus or up to eight minutes of local audio, choose a manual start or automatic repeated segment, listen to the excerpt, and inspect the prediction.
3. **Results & evidence:** inspect recorded model comparisons, predictions, target checks and uncertainty intervals.

Uploaded-audio inference is exploratory. The upstream source audio and exact library environment are unavailable, and our automatic selector is a transparent repetition heuristic rather than the source's pychorus implementation. The app states this limitation. It outputs a model score, not a calibrated probability of future commercial success.

## Original V1 results

- Dataset: **751 songs, 71 artist names, 518 audio features**.
- Labels: **366 year-end-hit / 385 other-chart-song**.
- Training: **597 songs, 52 artists**. Test: **154 songs, 19 different artists**.
- Five artist-disjoint folds inside training select models by **balanced accuracy**.
- Selected model: **Polynomial SVM** (chosen by training CV, before test evaluation).
- CV balanced accuracy: **53.6%**.
- Test balanced accuracy: **46.6%**; majority baseline: **50.0%**.
- Test accuracy: **46.8%**; positive-class F1: **0.406**; ROC-AUC: **0.456**.
- Approximate 95% interval for test balanced accuracy: **38.7%-55.8%**, using 2,000 bootstrap resamples of whole test artists.

**Conclusion:** these data and chorus features do not show a reliable predictive advantage for unfamiliar artists. A functional pipeline can produce a scientifically useful negative result. Do not claim high accuracy or that the experiment proves a catchy chorus causes popularity.

Some other candidates have higher test scores. They were not selected after looking at the test set. Their results are included as descriptive comparisons.

## Reproduce the experiment

From the project root with the environment activated:

```bash
python -m chorus_hit.train --output-dir work/v1_reproduction_new
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/build_reports.py --run-id v2_nested_001 --out work/reports_new --historical results/evaluations/v2_nested_001_historical
```

The fixed seed is 42. Training evaluates nine candidates: a majority baseline, logistic regression, LDA, linear/RBF/polynomial SVM, random forest, gradient boosting, and a 64/32-unit neural network. Model settings are recorded in `results/metrics.json`, and every CV trial appears in `results/cross_validation.csv`.

Preprocessing is inside each scikit-learn Pipeline. Imputation, constant-feature filtering, scaling, and optional 95%-variance PCA fit only on training folds. Trees use the original feature space. Artist, title, path, label, and track ID never enter the predictor. The saved model remains trained on the training partition only, preserving the validity of the held-out demo.

CPU training takes roughly a minute on the machine used to prepare this project; it can take longer elsewhere. No GPU or paid service is required. The pinned environment is recorded in `requirements.txt` and `environment-lock.txt`.

To restore the prepared data from its pinned public source:

```bash
python scripts/prepare_data.py
```

The importer removes source paths, renames metadata fields, adds stable IDs, checks the 518-column feature order, and records SHA-256 hashes in `data/provenance.json`. It does not execute downloaded code.

Command-line inference:

```bash
python -m chorus_hit.predict --track-id CH0001
python -m chorus_hit.predict --audio /path/to/song.wav --start 45
```

The first command reports whether the chosen track was held out. Use IDs from `results/test_predictions.csv` for a genuine test demonstration. Omit `--start` for automatic repeated-segment selection.

## Project files

| File / folder | Purpose |
| --- | --- |
| `app.py` | Interactive demo |
| `chorus_hit/data.py` | Data checks and grouped splits |
| `chorus_hit/train.py` | CV, model selection, holdout evaluation, persistence |
| `chorus_hit/audio.py` | Segment selection and 518 audio features |
| `chorus_hit/predict.py` | Command-line inference |
| `chorus_hit/plots.py` | Figures from measured results |
| `data/chorus_features.csv` | Prepared real-song feature table |
| `data/provenance.json` | Source, license, hashes, label limitations |
| `models/selected_model.joblib` | Model trained locally on training data only |
| `results/` | CV trials, split manifest, metrics, predictions, figures |
| `notebooks/Project_Walkthrough.ipynb` | Executed explanation and result inspection |
| `docs/Project_Report_2_Pages.pdf` | Main assignment write-up |
| `docs/Project_Summary_1_Page.pdf` | Alternate length for the conflicting heading |
| `docs/Project_Presentation.pptx` | Editable presentation with speaker notes |
| `docs/START_HERE.md` | Short run and study guide |
| `docs/VIVA_AND_DEMO.md` | Explanation, rehearsal, and likely questions |
| `tests/` | Checks for data integrity, leakage, metrics, inference, and app behavior |

The assignment heading says one page while its bullet says two pages. Both versions are supplied so the required one can be used without rewriting the project.

## Data and feature details

The 11 feature families have 74 total channels: chroma STFT/CQT/CENS (12 each), MFCC (20), RMS (1), spectral centroid (1), bandwidth (1), contrast (7), rolloff (1), tonnetz (6), and zero-crossing rate (1). Each contributes skewness, minimum, maximum, standard deviation, mean, median, and kurtosis: **74 × 7 = 518**.

The original CSV uses `kew` as the skewness column spelling. We retain it for compatibility. We also deliberately reproduce the upstream zero-crossing call's 22,050-sample frame length. Changes in library versions and unavailable original recordings mean exact waveform-to-feature equivalence is unverified. A fresh dataset should use one documented extraction configuration for all training and inference audio.

Artist grouping uses normalized artist strings. Collaborations and alternate names are not fully resolved. There is no release-date field suitable for a temporal holdout. Exact duplicate feature rows and artist/title pairs are rejected.

## Attribution

- Reference: Eric Liu, [Predicting Hit Songs Using Repeated Chorus](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf), CS229, 2021.
- Dataset: [Antonios Malak's repository](https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus), commit `838e76f96f7aa755a882b4d2581afe1e0ad3f8a9`. Source license retained in `licenses/DATASET_APACHE_2_0.txt`; see `NOTICE`.
- Method guidance: [scikit-learn on pipelines and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- Implementation and explanatory materials were prepared with OpenAI Codex assistance. Both team members should run and understand the work before presenting it.

No source audio or copyrighted recordings are redistributed in this repository. No faculty or TA access has been configured because those recipients were omitted at the user's request.
