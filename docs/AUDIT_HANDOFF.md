# Audit and experiment handoff — 2 October 2026

**Students:** Dhanush Sai Suprapadha (PES2UG24CS154) and Deepthi V (PES2UG24CS150).

The supplied brief was followed in order: preserve and reproduce V1; run the declared development experiments; implement and audit audio interfaces; perform the explicit historical comparison; integrate and verify the demo and deliverables. All work possible with the available feature data is complete. **The strict >75% target was not achieved. No fresh-test result exists.**

## 1. Status

| Category | Work and evidence |
| --- | --- |
| Implemented and tested | Exact V1 reproduction; actual-source hashing; alternate-import provenance and stable IDs; versioned datasets and immutable run artifacts; schema/label/hash validation; grouped quick and nested experiments; retained predictions, diagnostics and uncertainty; historical comparison; app/CLI/report integration; executed notebook and reviewed PDF/PowerPoint exports. |
| Implemented and tested with software fixtures | Audio validation/shared extraction, connected identity grouping, optional local MERT adapter/cache, worker checkpoint/resume and timeout checks. Synthetic fixtures contribute no real-song performance evidence. |
| Implemented but not run in the target setting | GitHub Actions workflow (no remote run); full CLI process-group forced-timeout path added after the recorded experiments (help/worker paths and checkpoint logic tested, no deliberately killed full study); real MERT inference and real-song ingestion (see blockers). |
| Blocked | No eligible original recordings, recording rights/versions/canonical-credit evidence, complete label/time verification, reviewed local MERT code/weights or genuinely fresh evaluation collection. Waveform-to-legacy-CSV parity is unverified. |
| Not attempted | Multi-segment aggregation, threshold tuning, calibration, ensembles, optional external boosting packages, temporal forecasting task, repeated favorable-seed search, promotion, remote push/merge/publication, and native PowerPoint application testing. |

## 2. Measured results

Both historical columns use exactly the same **154 songs / 19 normalized artist-name groups**, with 78 class-0 and 76 class-1 songs. The task remains **year-end hit versus other chart song**; both classes contain charting songs. The nested estimate evaluates the selection procedure on the original **597 development songs / 52 groups**, with 307 class-0 and 290 class-1 songs. It is a separate evaluation, not a matched historical before/after column.

| Metric | V1 historical | V2 historical | V2 nested procedure | V2 exceeds 75%? |
| --- | ---: | ---: | ---: | --- |
| Accuracy | 46.75% | 55.19% | 52.93% | No |
| Balanced Accuracy | 46.63% | 55.16% | 52.85% | No |
| Precision | 45.16% | 54.79% | 51.60% | No |
| Recall | 36.84% | 52.63% | 50.00% | No |
| F1 | 40.58% | 53.69% | 50.79% | No |

**Historical balanced-accuracy change: +8.54 percentage points.** The paired approximate 95% interval is **−1.59 to +16.86 points**, which includes zero. These data do not establish a reliable improvement.

Approximate 95% intervals from 2,000 resamples of whole artist-name groups:

| Metric | V2 nested procedure | V2 historical |
| --- | ---: | ---: |
| Accuracy | 48.33–57.39% | 47.01–61.94% |
| Balanced Accuracy | 48.26–57.34% | 47.13–61.51% |
| Precision | 43.60–59.43% | 40.00–67.93% |
| Recall | 43.69–56.13% | 37.84–63.54% |
| F1 | 44.35–56.32% | 40.60–63.28% |

These intervals condition on the recorded predictions and do not capture all training/selection uncertainty. There are only 19 historical groups. Fold standard deviation is not a confidence interval. V2 historical ROC-AUC is 0.5445; pooled nested ROC-AUC is deliberately absent because different outer models can have different score scales. Per-fold AUC is retained.

The quick run's best mean 3-fold **tuning** balanced accuracy was 52.88%; its pooled tuning OOF value was 52.84%. The final full-development selection also chose `lr_mi_50`: logistic regression, C=0.1, with 50 mutual-information-selected features from 518 raw inputs. The nested score above evaluates the complete selection procedure, not one final fitted model. No historical labels chose representations, models or thresholds. Fresh-test status is **unavailable / null**, not passed.

## 3. Confirmed findings and repairs

- **Original evidence reproduced:** full disposable V1 training matched all nine model metrics to 1e-12 and all 154 historical predictions. Original data, fitted model, scores, exports, notebook and attribution remain byte-identical. See `results/baseline/v1_manifest.json`, `results/audit/preservation_check.json`, and `results/audit/v1_reproduction_comparison.json`.
- **Source identity fixed:** `chorus_hit/data.py` and `audit.py` identify the actual loaded file and a separate processed frame fingerprint. `scripts/prepare_data.py` verifies exact legacy source bytes or requires explicit alternate provenance; alternate IDs remain stable when rows reorder. Imports do not overwrite datasets.
- **Run integrity fixed:** `chorus_hit/artifacts.py` validates run hashes, software versions, raw feature order, labels, predictions and memberships. App, CLI and report builders use the same loader. The app cache key includes the run manifest identity. Missing/extra/reordered/nonfinite inputs are rejected explicitly.
- **Experiment isolation:** `train_v2.py`, `estimators.py` and `evaluation.py` provide bounded, recorded grouped searches with fold-local learned transforms, eligible dummy baseline, deterministic tie handling, prediction reload checks and checkpoints. `train.py` now requires a new output directory. Original split safeguards were preserved.
- **Reports reflect the selected run:** `reporting.py`, `scripts/build_reports.py`, `build_notebook.py` and `build_presentation.mjs` derive output content from the recorded run. Tests also exercise a different model name/schema and a different conclusion, so the narrative is not frozen to V1.
- **Audio interfaces are explicit:** `audio.py` preserves `legacy-librosa-518-v1`; `shared-librosa-518-v2` uses a separate 2048-sample zero-crossing configuration. New extraction requires matched re-extraction/retraining. `ingestion.py` and `embeddings.py` enforce evidence/cache contracts. No new audio model was trained.

Remaining hypotheses: feature signal, inherited labels, recording/extractor differences, group coverage and overfitting may affect performance. Learning curves show train/validation gaps (about 81.5/49.5%, 70.1/51.4%, and 65.1/52.9% balanced accuracy as group coverage increases), but they do not establish one cause. Canonical performer identities remain unresolved; current scores use the original normalized strings. Near-duplicate flags do not silently alter groups or labels.

The six-record label sample was selected deterministically. Three positive labels have secondary MusicBrainz corroboration; three negative labels remain unresolved. Official chart pages returned HTTP 402. No full primary label verification or source recording verification is claimed. Evidence: `results/audit/label_sample_evidence.json` and `data/recording_audit_v1.json`.

## 4. Execution, tests, dependencies and logs

All Python experiments used the isolated **Python 3.12.14** environment and unchanged exact pins in `requirements.txt` / `requirements-dev.txt`; the baseline `environment-lock.txt` is preserved. Main packages: scikit-learn 1.9.1, NumPy 2.5.3, pandas 2.3.3, SciPy 1.18.1, librosa 0.11.0, Streamlit 1.64.0, joblib 1.6.0. The environment installation and `pip check` succeeded. No GPU, paid service, torch/transformers install or model download was used.

The following are the experiment commands (using `.venv/bin/python` from the project root unless noted):

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
python -m chorus_hit.train_v2 --config configs/v2_quick.json --run-id v2_quick_001
python -m chorus_hit.train_v2 --config configs/v2_thorough.json --run-id v2_nested_001
python -m chorus_hit.evaluate_run --run-id v2_nested_001 --historical --out results/evaluations/v2_nested_001_historical
python scripts/build_reports.py --run-id v2_nested_001 --out docs/v2 --historical results/evaluations/v2_nested_001_historical --date 2026-10-02
python scripts/build_notebook.py --run-id v2_nested_001 --out notebooks/v2/Project_Walkthrough.ipynb
python -m pytest -q -m 'not integration'
python -m chorus_hit.predict --run-id v1_baseline --track-id CH0001
python -m chorus_hit.predict --run-id v2_nested_001 --track-id CH0001
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Existing run IDs and export destinations cannot be reused for new runs. See `REPRODUCE_V2.md` for safe new destinations. The original unmodified `python -m chorus_hit.train` was executed only in the disposable V1 archive; its exact absolute command and working directory are in `results/audit/v1_reproduction_command.json`. PDF wording was repaired in `../../work/pdf-v2/repaired` with the same report parameters, then the revised V2 exports were copied into `docs/v2/`. PowerPoint was built with the bundled Node/artifact-tool runtime, using `scripts/build_presentation.mjs`; its initial export failure and successful repair are retained.

| Execution | Result / measured duration | Evidence |
| --- | --- | --- |
| V1 full disposable reproduction | Exit 0; 49.76 seconds wall time | `v1_reproduction_command.json`, `v1_reproduction.log`, comparison JSON |
| Quick development | 114 completed trial records; 0 failed; 0 trial warnings; 16.39 seconds in runner | `v2_quick_001_console.log`, run manifest/trials |
| Nested development | 689 completed trial records; 0 failed; 0 trial warnings; 77.18 seconds in runner | `v2_nested_001_console.log`, run manifest/trials |
| Final full software suite | 20 passed, 4 dependency/decoder deprecation warnings; 51.39 seconds pytest / 54.38 wall | `final_tests.log`, `final_commands.json` |
| Fast suite | 12 passed, 8 integration tests deselected; 17.56 seconds pytest / 19.11 wall | `fast_tests.log`, `final_commands.json` |
| CLI / help checks | Both prediction commands and all recorded help commands exit 0 | `final_commands.json` and its named logs |
| Final display adjustment | App run-switch/prediction integration check and actual browser verification | `app_final_check_command.json`, `browser_verification.json` |
| Notebook | 13 cells, all 6 code cells executed, 0 errors | notebook and `export_manifest.json` |
| PDFs / presentation | 2-page and 1-page PDFs; 10-slide deck, 4 native tables, 2 native charts | `export_manifest.json`, `presentation_validation.json` |

All abbreviated log names above are under `results/audit/`. Exact command arrays, exit statuses and durations are retained in the command JSONs. The original test timing and package checks are in `phase_a_commands.json`. Durations were not captured for every export/metadata lookup; no duration is invented for those operations.

Real errors retained: initial presentation export rejected excessive literal numeric precision; the final chart displays one decimal while experiment records retain full precision. The final deck passed package/layout/font/native-chart checks. A local model metadata request timed out; an official web commit subsequently supplied the pinned MERT revision. Primary chart evidence remains inaccessible. No candidate training trial failed in either completed study.

All three PDF pages and all ten slide renders were visually reviewed. The final PowerPoint was imported and rerendered; all ten renders match the reviewed originals. Actual Microsoft PowerPoint rendering and remote CI remain unverified. Original six tests remain unchanged.

## 5. Files, branch and reproducibility

Local branch: **`audit/chorus-v2`**, based on `0e9e4628d0a72af79694bc555d64a562e83dcfa2`. **No push, merge or publication was performed.** The sibling package receipt records the final local commit and package hashes. The review patch is `../Chorus_Hit_Audit_V2.patch`; the complete package is `../Chorus_Hit_Audit_V2.zip`. The baseline archive is `../Chorus_Hit_Baseline_V1.zip`.

Change groups: provenance/data (`data.py`, `audit.py`, import scripts); evaluation (`train_v2.py`, `estimators.py`, `evaluation.py`, configs); artifacts/inference (`artifacts.py`, `evaluate_run.py`, `predict.py`, `app.py`); audio (`audio.py`, `ingestion.py`, `embeddings.py`); exports (`reporting.py`, builders, `docs/v2`, `notebooks/v2`); evidence (`results/audit`, `results/baseline`, `results/v2`, `results/evaluations`); tests, CI and documentation. See the patch for every file.

The experiment manifests accurately record the original commit with a dirty working tree and executed code hash. The exact modules that produced the two runs are preserved in `results/audit/v2_executed_code.zip` and its verified identity JSON. Later reporting/audio/supervision changes did not rewrite measured runs. A final clean commit does not retroactively become their executed source.

Start here:

- `docs/v2/Project_Report_2_Pages.pdf` — updated full assignment report.
- `docs/v2/Project_Summary_1_Page.pdf` — alternate length for the assignment's conflicting page instructions.
- `docs/v2/Project_Presentation.pptx` — editable slides and speaker notes.
- `notebooks/v2/Project_Walkthrough.ipynb` — executed walkthrough.
- `docs/VIVA_V2.md` — explanation, rehearsal and likely questions.
- `results/audit/session_manifest.json` — index of execution/evidence/status.

The default demo run remains **`v1_baseline`** in `configs/active_run.json`. V2 is an optional research view in the sidebar. It has not been promoted. The demo was verified locally at `http://127.0.0.1:8501/`; it is not publicly hosted.

## 6. Concrete missing inputs and next action

For each recording: a permitted local audio file; rights basis; checksum; recording/version identity; canonical credited performers and evidence; duplicate links; verified target label and chart coverage; release and chart evidence dates. Apply one extraction and segment policy consistently to both classes. For MERT, add reviewed local code/config/weights at the pinned revision and compatible optional dependencies. For confirmation, collect a genuinely new evaluation set and lock inclusion/grouping/label rules before selecting another final model. Repartitioning the already examined 751 songs cannot create freshness. See `AUDIO_DATA_REQUIREMENTS.md` and the null fields in `data/recording_audit_v1.json`.

**Next unblocked action:** review this package and rehearse the demo with `VIVA_V2.md`. The next research step is completing the recording and label evidence; another search over the same data is not justified as a route to a guaranteed score.

## 7. Viva explanation

We turned each chorus into numerical sound measurements, compared models with artists kept separate, and tested the entire model-selection process using nested folds. The selected candidate uses 50 statistically selected measurements and logistic regression. Its results are slightly above 50%, with uncertainty that includes chance, and the 75% target remains unmet. The project now offers reproducible experiments, a working demo and an honest negative result. It does not prove that chorus properties cause commercial success or that uploaded music can reliably be classified. Both students should run and understand the project; implementation and materials were prepared with Codex assistance.
