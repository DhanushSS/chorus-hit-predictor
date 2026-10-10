# Approved HSP-S feature experiment

The user approved this separate fallback on 10 October 2026 after the exact-278
audio acquisition attempt was blocked. It does not replace the original dataset,
the active Chorus Hit Predictor model or the 278-song metadata project.

## Source and attribution

Michael Vötter, Maximilian Mayerl, Günther Specht and Eva Zangerle (2021),
*Novel Datasets for Evaluating Song Popularity Prediction Tasks*,
[Zenodo DOI 10.5281/zenodo.5383858](https://zenodo.org/records/5383858).
The publisher declares CC BY 4.0. This implementation uses
`hsp-s_acousticbrainz.parquet` (210,697,324 bytes), pinned publisher MD5
`c3372acb249f9243e92613537825ad4d`. The entire file must pass the checksum;
partial downloads and sparse inspection files are not training data.

The [authors' paper](https://dbis-informatik.uibk.ac.at/sites/default/files/2022-06/ISM_2021__Hit_Song_Prediction.pdf)
describes matching Billboard Hot 100 entries from 11 August 1958 through 6 July
2019 to the Million Song Dataset, then sampling non-hits from MSD. The released
file has 7,736 rows: 3,870 have chart fields and 3,866 lack them. Labels here are
the **publisher benchmark's chart association**, not independently certified
absence from the charts, and not the current 278-song cohort's 2023 cutoff.

No recordings are downloaded. The source uses whole-recording Essentia descriptors,
not our 15-second librosa chorus vectors. Raw data remains in ignored local storage.
The release-year CSV has 44,272 rows and cannot be joined by position; this run does
not use it. No listening counts, popularity scores or publication-year guesses
are used as predictors.

## Preregistered feature and evaluation policy

Configuration: `configs/hsp_s_acousticbrainz.json`, saved before any model fitting.

- Only the source schema's **436 numeric scalar** `lowlevel`, `rhythm`, and `tonal`
  descriptors are eligible as predictors. Vector/covariance fields and high-level
  pretrained genre/mood classifications are outside this first experiment.
- Compare all scalar descriptors, a compact view of mean/variance and singleton
  descriptors, and fold-local PCA retaining 95% training variance. These are HSP-S
  representations; they are not named or represented as 518/std74 chorus features.
- Compare the existing model families in **27 analogous configurations**: dummy,
  logistic regression, linear/RBF SVM, random forest, extra trees, small MLP and
  histogram boosting. Hyperparameters are fixed in the new config.
- Median imputation, constant-feature removal, scaling and PCA fit on training
  folds only. No row-random neural early stopping. Native class thresholds remain
  fixed; probability-like outputs are uncalibrated scores.
- Rows with inconsistent chart fields, unusable artist identity or no finite audio
  descriptors are excluded with reasons. Identical vectors are deduplicated by
  UUID order; conflicting labels for identical vectors/recordings are excluded.
- Connect recording-artist IDs and normalized names, recording IDs, album IDs,
  title/credit work proxies, encoded-audio MD5 and identical features before
  partitioning. Guest links propagate transitively. These checks rely on source
  tags; incomplete tags, aliases and covers remain limitations.
- One fixed **20% connected-group holdout**, seed **20261008**, with at least five
  independent groups per class in both partitions. No seed retries.
- **Five outer / three inner grouped folds** on development data. Inner mean
  balanced accuracy selects the model; candidate ID breaks exact ties. Outer
  predictions estimate the whole selection procedure. Final selection uses only
  development data, followed by one full-development fit.
- Individual fits run in supervised subprocesses with a 60-second cap. Failed and
  timed-out fits remain in the comparison; a candidate needs all its inner fits.
  The declared trial-fit budget is 32,400 seconds. Checkpoints are resumable only
  with unchanged data, config, code and environment.
- Development-only nuisance and within-group label-permutation checks run before
  locked evaluation. The predeclared nuisance threshold (0.65 balanced accuracy)
  and suspicious-permutation flags block the final test for review. No automatic
  cohort revision or test opening is used to bypass a flag.
- Final test access is recorded exclusively **before loading test features/labels**.
  Failure does not refund access. Accuracy, balanced accuracy, positive and macro
  F1, precision, recall, per-class support, ROC-AUC, AP, confusion matrix and a
  connected-group bootstrap interval are reported if evaluation is permitted.

## Reproduction

Use the project's pinned Python 3.12 environment from the repository root:

```bash
python -m chorus_hit.hsp_s download --out output/hsp_s/source
python -m chorus_hit.hsp_s prepare \
  --source output/hsp_s/source/hsp-s_acousticbrainz.parquet \
  --out output/hsp_s/split_001
python -m chorus_hit.hsp_s train \
  --split output/hsp_s/split_001 --out results/hsp_s/run_001
python -m chorus_hit.hsp_s evaluate \
  --split output/hsp_s/split_001 --run results/hsp_s/run_001
```

Use `train --resume` only for an interrupted unfinished run. Completed outputs
cannot be overwritten. Never remove a test-access ledger to evaluate again.

Streamlit's separate **Hit vs Non-Hit feature study** page displays only a compatible,
evaluated HSP-S run. It accepts the exact ordered feature CSV schema (1–100 records).
It cannot use the original librosa upload extractor for this model. The original
audio-upload model remains on the main page.

Reported results must identify HSP-S as a separate retrospective benchmark, explain
its sampling and label limitations, and avoid numerical superiority claims against
Eric Liu or the original model on different datasets and evaluation protocols.
