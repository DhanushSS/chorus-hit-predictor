# Predicting hit songs from a repeated chorus

Machine learning mini-project by Dhanush Sai Suprapadha (PES2UG24CS154) and Deepthi V (PES2UG24CS150).

We test whether measurements from a 15-second chorus distinguish songs in a Billboard year-end Hot 100 collection from other songs that appeared on weekly Hot 100 charts. Both classes contain charted songs. The dataset has 751 songs and 518 audio features per song.

## Run

Use Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

On Windows, activate `.venv\Scripts\activate`. The app starts with the original baseline model. Choose the V4 study in the sidebar to inspect the latest experiment. Uploaded audio is exploratory: the original recordings and exact feature-extraction environment are unavailable.

## Current result

| Artist-separated development evaluation | V3 | V4 |
| --- | ---: | ---: |
| Accuracy | 53.27% | 54.61% |
| Balanced accuracy | 53.25% | 54.66% |
| F1 | 52.31% | 54.76% |

V4 uses 74 chorus-variation features within the saved 518-feature input pipeline. Its paired balanced-accuracy increase over V3 is 1.41 percentage points, with an approximate 95% interval of -4.29 to +7.33 points. The interval includes zero, so the improvement is uncertain. The original 154-song test partition was examined in earlier work and was not reused to select V4. A new, independently checked collection is needed to evaluate generalization.

## Compute efficiency

A local, single-thread benchmark fits three fixed pipelines on the same five artist-separated development folds. It times only training and prediction, not feature extraction:

| Pipeline | Exploratory accuracy | Median fit per fold | Predict 154 songs | Saved pipeline |
| --- | ---: | ---: | ---: | ---: |
| V4 logistic regression, 74 features | 56.6% | ~6 ms | ~0.6 ms | 6.6 KiB |
| MLP neural-network control with PCA | 50.6% | ~260 ms | ~1.4 ms | 1040 KiB |
| Random-forest control | 51.1% | ~400 ms | ~7 ms | 479 KiB |

Run `python -m scripts.benchmark_models` to repeat the comparison locally. These are measurements on one computer and our dataset, not timings for Eric Liu's implementation. The fixed V4 setting was chosen after earlier development work, so its benchmark accuracy is exploratory and does not replace the nested selection result above. Paired artist-group bootstrap intervals for accuracy and precision versus both controls include zero.

`app.py` and `chorus_hit/` contain the app and ML pipeline. `data/` holds the feature table and provenance; `configs/` holds experiment settings; `results/v2/` holds the saved baseline and V2–V4 models, predictions and verification manifests. The other small files in `results/` support the original baseline and historical comparison. Reports and slides are kept outside this code repository.

Check the saved models and software with `python -m pytest -q`. For a labelled example, run `python -m chorus_hit.predict --run-id v4_std74_001 --track-id CH0001` and inspect whether that song was in training or held out.

The features were adapted from the [repeated-chorus dataset](https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus) under Apache-2.0; see `NOTICE`, `licenses/`, and `data/provenance.json`. Our class definition differs from [Eric Liu's study](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf), so its accuracy is not directly comparable.

## Ceiling check: can any model reach 70%+?

`scripts/leakage_demo.py` re-evaluates five model families with repeated 5-fold CV on all 751 songs, once with a leaky random split and once with artist-disjoint folds. Results are in `results/leakage_demo.csv`.

- Best artist-disjoint balanced accuracy: **53.9%** (Extra trees). Best random-split score: **54.9%**.
- Even when the same artists appear in train and test, no model gets near 70%. The 518 chorus statistics carry very little signal for this label, so tuning cannot fix it.
- Higher numbers would need different information (pretrained audio embeddings, more songs, or a charted vs never-charted label), not a better classifier.
