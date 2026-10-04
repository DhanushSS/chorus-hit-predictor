# Predicting Hit Songs Using Repeated Chorus

**UE24CS352A Machine Learning mini-project**

| Student | USN |
| --- | --- |
| Dhanush Sai Suprapadha | PES2UG24CS154 |
| Deepthi V | PES2UG24CS150 |

Can measurements from a 15-second chorus help distinguish a Billboard year-end hit from another song that appeared on the weekly Hot 100? This project tests that question with artist-separated machine-learning evaluation. **Both classes contain charted songs.** The model does not establish that a chorus causes a song to become popular.

## Run the project

Use Python 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

On Windows, activate `.venv\Scripts\activate` and use `run_demo.bat` if preferred. The app opens at `http://localhost:8501`. It starts with the original baseline model; select **V4 chorus-variation study** in the sidebar to inspect the latest research model. Audio uploads are exploratory because the original training recordings and exact extraction setup are unavailable.

## Data and method

The inherited feature table has **751 songs**, **71 normalized artist groups**, and **518 measurements** of timbre, pitch, energy and spectral structure. The original development partition has **597 songs**; **154** songs from different artists form a historical test partition. Names and titles are grouping/identification fields, never predictor inputs.

The latest study retained 55 earlier model settings and added eight logistic-regression settings using **74 chorus-variation features**. Five outer artist-separated folds estimate the selection procedure; three inner folds choose settings without using each outer validation fold. Imputation, scaling and feature selection are fitted only on training folds. The 154-song historical partition was not evaluated again for V4.

| Development result | V3 | V4 |
| --- | ---: | ---: |
| Ordinary accuracy | 53.27% | **54.61%** |
| Balanced accuracy | 53.25% | **54.66%** |
| F1 score | 52.31% | **54.76%** |

V4's paired balanced-accuracy change is **+1.41 percentage points**, with an approximate 95% interval of **-4.29 to +7.33 points**. The interval includes zero, so this is an exploratory improvement rather than reliable confirmation. The five-metric 75% target is unmet. A genuinely new collection with verified recording and chart identities is still needed.

## Project files

| Location | Purpose |
| --- | --- |
| `app.py`, `chorus_hit/` | Demo, data checks, feature extraction, model loading and evaluation |
| `data/chorus_features.csv`, `data/provenance.json` | Licensed input features and source/label limitations |
| `configs/` | Recorded experiment settings and active-model choice |
| `results/v2/` | Validated runs, predictions, models and compact trial archives |
| `docs/Project_Report_2_Pages.pdf` | Current two-page assignment write-up |
| `docs/Project_Summary_1_Page.pdf` | One-page version, supplied because the assignment gives conflicting lengths |
| `docs/V4_STD74_RESULTS.md` | Detailed result and uncertainty discussion |

The required presentation file is retained in `docs/` for later preparation. The complete earlier research history remains available in Git; individual training-trial JSON files are bundled in `trials.zip` within each completed run. The original per-trial hashes remain in each run's manifest and are checked whenever a run is loaded.

## Check the saved results

```bash
python -m pytest -q
python -m chorus_hit.predict --run-id v4_std74_001 --track-id CH0001
```

The first command checks software and saved evidence. The second produces a labelled example prediction; inspect its training/test status before interpreting it as an evaluation case. Results can also be explored in the app. The latest tested suite passed **30 tests**. To rebuild the PDFs with the validated saved results, run `python scripts/build_submission_pdfs.py`.

The data were adapted from the [Predicting Hit Songs Using Repeated Chorus feature collection](https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus) under Apache-2.0; see `NOTICE`, `licenses/`, and `data/provenance.json`. The class definition in this project differs from [Eric Liu's reference report](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf), so their accuracy numbers are not a controlled comparison.
