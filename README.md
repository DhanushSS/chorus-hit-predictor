# Predicting hit songs from a repeated chorus

Machine learning mini-project by Dhanush Sai Suprapadha (PES2UG24CS154) and Deepthi V (PES2UG24CS150).

We test whether measurements from a 15-second chorus distinguish songs in a Billboard year-end Hot 100 collection from other songs that appeared on weekly Hot 100 charts. Both classes contain charted songs. The dataset has 751 songs and 518 audio features per song.

## Setup and existing demo

```bash
git clone https://github.com/DhanushSS/chorus-hit-predictor.git
cd chorus-hit-predictor
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
python -m chorus_hit.readiness
python -m streamlit run app.py
```

Python **3.12** and the exact dependency pins are required by the saved models.
Initial cloning/package installation needs internet. Saved-model prediction,
training from the checked-in feature table and the app work offline; no account,
Spotify connection, GPU, model download or Hugging Face token is needed.

| Platform | Saved-run app/prediction | Supervised V2–V4 training and diagnostics |
| --- | --- | --- |
| macOS | Locally tested | Supported and tested |
| Linux | CI environment | Supported; CI timeout tests |
| Windows | Not tested; requires compatible package pins/codecs | Unsupported; fails early with an explanation |

On Windows the venv activation command differs; there is no tested Windows training
promise. Package guards must not be bypassed. The app starts with `v1_baseline`.
Choose `v4_std74_001` in the sidebar to inspect V4 without changing the active model.
Uploaded audio is experimental: source waveform equivalence and new-song accuracy
are unverified. See [audio contract](docs/audio_contract.md).

## Current result

| Normalized artist-name-grouped development evaluation | V3 | V4 |
| --- | ---: | ---: |
| Accuracy | 53.27% | 54.61% |
| Balanced accuracy | 53.25% | 54.66% |
| F1 | 52.31% | 54.76% |

V4 uses 74 chorus-variation features within the saved 518-feature input pipeline. Its paired balanced-accuracy increase over V3 is 1.41 percentage points, with an approximate 95% interval of -4.29 to +7.33 points. The interval includes zero, so the improvement is uncertain. The original 154-song test partition was examined in earlier work and was not reused to select V4. A new, independently checked collection is needed to evaluate generalization.

## Compute efficiency

A local, single-thread benchmark fits three fixed pipelines on the same five normalized artist-name-grouped development folds. It times only training and prediction, not feature extraction:

| Pipeline | Exploratory accuracy | Median fit per fold | Predict 154 songs | Saved pipeline |
| --- | ---: | ---: | ---: | ---: |
| V4 logistic regression, 74 features | 56.6% | ~6 ms | ~0.6 ms | 6.6 KiB |
| MLP neural-network control with PCA | 50.6% | ~260 ms | ~1.4 ms | 1040 KiB |
| Random-forest control | 51.1% | ~400 ms | ~7 ms | 479 KiB |

To explicitly rerun this diagnostic into a new directory, use `python -m scripts.benchmark_models --out output/benchmarks/new_run_001 --timeout-seconds 120`. Existing directories are refused. These are measurements on one computer and our dataset, not timings for Eric Liu's implementation. The fixed V4 setting was chosen after earlier development work, so its benchmark accuracy is exploratory and does not replace the nested selection result above. Paired artist-group bootstrap intervals for accuracy and precision versus both controls include zero.

### Final accuracy experiment — 2026-10-06

One predeclared equal-weight ensemble of the eight existing V4 std74 logistic
settings was evaluated on the same 597 development songs and five artist-name
grouped folds. There was no threshold, seed or weight search. The fixed V4 control
reproduced **56.62% accuracy / 54.86% precision / 60.34% recall**; the ensemble
reached **55.28% / 53.61% / 58.97%**, respectively. Its accuracy difference was
-1.34 percentage points (paired group-bootstrap 95% interval: -3.82 to +0.84).
The ensemble was not promoted. Accuracy experimentation is now stopped at the
user's request. This exploratory check does not replace the **54.61% nested V4
assessment** or establish new-song accuracy. Code: `scripts/final_accuracy_check.py`;
protocol and evidence hashes are in [the exposure registry](docs/evaluation_registry.json).

`app.py` and `chorus_hit/` contain the app and ML pipeline. `data/` holds the feature table and provenance; `configs/` holds experiment settings; `results/v2/` holds the saved baseline and V2–V4 models, predictions and verification manifests. The other small files in `results/` support the original baseline and historical comparison. Reports and slides are kept outside this code repository.

Check the saved models and software with `python -m pytest -q`. For a labelled example, run `python -m chorus_hit.predict --run-id v4_std74_001 --track-id CH0001` and inspect whether that song was in training or held out.

The features were adapted from the [repeated-chorus dataset](https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus) under Apache-2.0; see `NOTICE`, `licenses/`, and `data/provenance.json`. Our class definition differs from [Eric Liu's study](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf), so its accuracy is not directly comparable.

## Exploratory split diagnostic

The preserved `results/leakage_demo.csv` compares five model families using five
repeats of five-fold CV on **all 751 songs**, including the historical partition.
The protocols are a random song split with artist overlap and a normalized
artist-name-grouped split. Its highest recorded grouped balanced accuracy is
53.9% (Extra trees), versus 54.9% for the highest random-song result.

In the evaluated models and settings, repeated artist-name-grouped evaluation did
not reach 70% balanced accuracy. This does **not** establish a universal performance
ceiling. Further data, representation or modelling changes require separate
evaluation. This retrospective experiment does not validate future commercial
success prediction or calibrated success probabilities.

The diagnostic is optional, not a routine development/test step. Its explicit CLI is:

```bash
python -m scripts.leakage_demo --out output/diagnostics/new_run_001 --repeats 5 --seed 0 --workers 1 --timeout-seconds 300
```

It records every split, warnings, failures, unrounded values, code/data identity
and the all-data exposure. Outputs are new directories, not replacements for the
old CSV. Do not rerun it to hunt for favorable seeds. It was **not rerun on the
full dataset during remediation**; software tests use synthetic fixtures.

## Reproduce and verify

Use a new output/run name for each reproduction. Existing completed runs and the
active baseline are immutable. These steps do not imply that rerunning inspected
songs creates a fresh test.

```bash
# Fast integrity/software checks; no full search.
python -m pytest -q
python -m chorus_hit.artifacts
python -m chorus_hit.predict --run-id v4_std74_001 --track-id CH0001
python -m chorus_hit.audit --out output/audits/my_audit_001
python -m chorus_hit.readiness --evidence-only --out output/audits/my_evidence_001.json

# Optional full reproductions, macOS/Linux; original outputs remain intact.
python -m chorus_hit.train --output-dir output/reproductions/v1_my_repro_001
python -m chorus_hit.train_v2 --config configs/v4_std74.json --run-id v4_my_repro_001
```

`CH0001` is a development/training row for the V4 final model; its CLI result is an
interface demonstration, not held-out accuracy. V1 reproduction creates a new
model/metrics/split/figures under the supplied directory. V4 produces
`results/v2/v4_my_repro_001/` with model, predictions, fold/trial records, summary
and verification manifest. Neither command promotes a model or modifies
`configs/active_run.json`.

Only if `results/v2/.v4_my_repro_001.partial/` exists and code/config/data match:

```bash
python -m chorus_hit.train_v2 --config configs/v4_std74.json --run-id v4_my_repro_001 --resume
```

Resume reuses completed matching checkpoints. The wall budget is **per invocation**;
each explicit resume receives that configured budget, while elapsed totals remain
telemetry. Each running fold fit separately enforces `fit_timeout_seconds`; queued
waiting is excluded. The outer process supervisor also bounds final fitting and
bootstrap work. Final refitting has the invocation limit, not a separate fold-fit
limit. Timeout/interrupt cleanup preserves completed checkpoints. A code change
invalidates partial-run identity; choose a new ID rather than modifying old state.

Full V1/V4 training was not rerun merely for this software remediation: no labels,
features, hyperparameters, thresholds, grouping or scientific metric definitions
changed. CLI help and bounded training fixtures were checked. Exact new timing
results will vary by machine, and a new code identity is recorded for reproductions.

## Architecture and evidence map

| Location | Responsibility |
| --- | --- |
| `app.py`, `chorus_hit/predict.py` | Demo and CLI using one validated run loader |
| `chorus_hit/data.py`, `audit.py`, `ingestion.py` | Feature table, source identity, strict new-record contract |
| `chorus_hit/audio.py` | Segment selection and explicit legacy/shared extraction contracts |
| `chorus_hit/estimators.py`, `evaluation.py` | Fold-local preprocessing, classifiers, metrics, group bootstrap |
| `chorus_hit/train.py`, `train_v2.py`, `supervision.py` | Baseline and bounded/resumable development experiments |
| `chorus_hit/artifacts.py`, `contracts.py` | Hash, environment and semantic result validation |
| `configs/active_run.json` | Original active run; selecting research in the UI does not edit it |
| `results/v2/v1_baseline`, `v2_nested_001`, `v3_nested_001`, `v4_std74_001` | Original completed results, not reproduction destinations |
| `scripts/benchmark_models.py`, `leakage_demo.py` | Optional exploratory diagnostics, never promotion/fresh-test evidence |
| `docs/` | Task alignment, exposure registry, evidence counts, remediation and rehearsal notes |

See [task approval and adaptation](docs/task_alignment.md),
[exposure registry](docs/evaluation_registry.json),
[evidence counts](docs/evidence_audit.json), and
[demo checklist](docs/demo_checklist.md). All labels remain inherited; all canonical
recording identities remain unresolved. The new-record review template is
[recording_review_template.json](docs/recording_review_template.json).

## Troubleshooting

- **Package mismatch:** use a clean Python 3.12 venv and pinned dependencies, then
  `python -m pip check`. Do not edit a saved manifest or disable its version guard.
  If installation fails, retain the resolver error for a versioned compatibility fix.
- **Missing/corrupt selected model:** restore the matching trusted repository files;
  the app stops prediction rather than silently switching models. Hash checks do
  not make arbitrary downloaded pickle/joblib files safe to load.
- **Missing optional V2 evidence:** the comparison panel reports unavailable; a
  valid selected run continues. Missing frozen baseline membership disables only
  the historical-song demo for an otherwise valid research run.
- **Rejected uploads:** check permissions, supported codec, finite non-silent mono
  audio, 15-second minimum, eight-minute maximum, and timestamp boundaries.
  Extraction schema/order/version must match the model.
- **Training platform/budget error:** use macOS/Linux and choose a deliberate
  per-invocation budget. Do not repeatedly resume without accounting for compute.
- **Missing intervals:** insufficient valid two-class bootstrap samples are reported
  as unavailable, never replaced by an invented confidence interval.

Repository visibility remains public at the user's instruction. PDFs/slides and
the actual review-device rehearsal are deferred handoff tasks. Rehearsal, Q&A
preparation and further faculty/access confirmation are outside the current work
list at the user's request; adaptation approval is already recorded.
