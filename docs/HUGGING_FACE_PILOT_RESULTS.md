# Hugging Face assisted feature pilot — 3 October 2026

## Measured result

On **340 development songs from 46 artist-name groups**, combining existing chorus features with published musicnn features reached **54.17% balanced accuracy**. The matched existing-feature baseline scored **50.79%**, and musicnn alone scored **52.09%**.

This is an **exploratory artist/title-linkage pilot**, not a validated improvement to the deployed chorus model. The combined gain is **+3.38 percentage points**, with an approximate paired 95% artist-bootstrap interval of **−0.22 to +7.21 points**. It includes zero. Recording versions, segment equivalence and external model pretraining overlap are unverified.

| Inputs | Accuracy | Balanced accuracy | Hit precision | Hit recall | Hit F1 |
|---|---:|---:|---:|---:|---:|
| Existing features, MI50 | 54.41% | 50.79% | 59.66% | 70.65% | 64.69% |
| Published musicnn features | 56.47% | 52.09% | 60.47% | 76.12% | 67.40% |
| Both combined | 57.35% | 54.17% | 62.07% | 71.64% | 66.51% |

There are 201 positive and 139 negative labels. An always-positive reference would achieve **59.12% ordinary accuracy but 50% balanced accuracy**. Thus none of these models beats that reference on ordinary accuracy. The isolated 76.12% recall does not meet the five-metric target; musicnn recognizes only 28.06% of negative examples. The combined model's balanced-accuracy interval is **49.00%–59.22%**. No arm meets all five strict >75% checks.

The previous **53.25% V3 result remains unchanged**. It used 597 development songs and a different selection procedure; comparing it directly with this 340-song score would confound the data subset and method. The pilot models are not offered in the demo because uploads cannot reproduce the external feature extraction or verify recording matches.

## What Hugging Face enabled

- Inspected the [Music4All mirror](https://huggingface.co/datasets/Leon299/music4all) at revision `a391160e3e17f351d5ab2d05439a7d3d7f0440eb`: 109,269 artist/title records. Normalization uses only Unicode NFKC, case folding and whitespace collapse. Unique matches were required in both sources. No fuzzy matching or title-only matching was used.
- Identified 434 candidate links in the full project table; **340 belong to the original development partition**. Historical rows were excluded before fitting. These links are candidates, not verified recording identities.
- Retrieved the companion authors' [Music4All-Onion release](https://zenodo.org/records/15394646), labelled CC BY 4.0, and verified the published MD5 for `id_musicnn.tsv.bz2`. Its 50 released audio-derived features were used locally. No raw audio or MERT model was downloaded, and no paid compute was used.
- The [Billboard Hub dataset](https://huggingface.co/datasets/willcb/billboard-hits) preview exposes title, year and performer, without audio. The [FMA dataset](https://huggingface.co/datasets/benjamin-paine/free-music-archive-small) exposes audio and individual licences but no matching Billboard year-end labels. Neither was used for training.
- The plugin's dataset-search endpoint previously reported disabled by server configuration. Public Hugging Face metadata and Dataset Viewer requests supplied the missing inspection capability.

Spotify popularity, listening counts, chart position, artist names, titles, release dates, and external Spotify acoustic metadata were **not predictor inputs**. The separately inspected metadata file is retained only as provenance and is not read by the pilot runner.

## Protocol and verification

The protocol and runner were committed before fitting at `6ce23201633608b1644f28f03f4845961f5b8911`. All three arms use logistic regression C=0.1 and its native threshold, with five original V3 outer folds restricted to the same linked development songs. Imputation, scaling and mutual-information selection are fitted only on training rows; combined inputs concatenate 50 selected legacy features and 50 external features. There was no hyperparameter search or threshold tuning. No historical scoring or fresh-test claim was made.

**15 fits completed; zero trial warnings or failures.** All 15 saved models were reloaded and reproduced their predictions and scores. Metrics were independently recomputed with scikit-learn; artist separation, development-only membership, source hashes and learned scaling means were checked. Exact duplicate external vectors crossing artist groups would be excluded; none occurred. These checks do not detect every alternate recording or performer alias.

The approximate paired bootstrap uses 2,000 whole-artist resamples of fixed predictions. It does not include all model-fitting uncertainty, correct for repeated inspection, or validate the recording links. The same development data were previously studied; these are exploratory findings. A single fixed protocol was run without adding settings after seeing scores.

Evidence is in `results/hf_features_001/` (inputs, fold membership, predictions, models, immutable manifest and summary) and `results/hf_feasibility_001/` (source records, checksums, verification and log). Downloaded bulk source files are excluded from Git but available locally, with pinned URLs and hashes in `source_manifest.json`. This new study is local only, following the user's private-study preference.

## Next useful step

Verify candidate recording identities and obtain permitted, matching chorus audio. Then compare a frozen MERT representation against legacy features extracted from those exact same segments, with a new test collection reserved before model selection. The [original Music4All authors](https://sites.google.com/view/contact4music4all) provide an access-request contact; no request has been sent. Access to generic audio alone does not resolve recording, chorus or chart-label verification.

## Attribution

Marta Moscati, Emilia Parada-Cabaleiro, Yashar Deldjoo, Eva Zangerle, and Markus Schedl. *Music4All-Onion — A Large-Scale Multi-faceted Content-Centric Music Recommendation Dataset*. CIKM 2022. https://doi.org/10.1145/3511808.3557656. Feature release: https://doi.org/10.5281/zenodo.15394646, CC BY 4.0. This pilot subsets and joins the released features; transformations are fitted separately within each training fold.

Music4All: Igor André Pegoraro Santana and colleagues, *Music4All: A New Music Database and Its Applications*, IWSSIP 2020. Metadata mirror supplied by Leon299 on Hugging Face; its card does not establish a separate licence or verify recording versions. Original chorus-table attribution remains in `NOTICE`.

## Reproduction

Use the existing Python 3.12 environment and pinned requirements. Run `python scripts/fetch_hf_pilot_sources.py` to restore and hash-check the three required research source files. The completed run directory is immutable: `python scripts/run_hf_feature_pilot.py` intentionally refuses to overwrite it. To independently repeat the declared run, use a disposable copy of the repository without `results/hf_features_001/`, then execute the runner. This preserves original evidence; there is no automatic model promotion.
