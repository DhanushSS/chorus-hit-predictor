# Setup and supported commands

Use Python 3.12 and the exact pinned requirements. All commands below run from the project root. No GPU, optional embeddings package or paid service is needed for the CSV experiments.

## Setup and quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
python -m streamlit run app.py
```

On Windows use `py -3.12 -m venv .venv` and `.venv\Scripts\activate`. The Mac and Windows launchers remain available. The default model is `v1_baseline`; `configs/active_run.json` records the deliberate no-promotion decision. Selecting the V2 research option in the sidebar does not change that pointer.

## Audit, quick exploration and thorough evaluation

```bash
python -m chorus_hit.audit --data data/chorus_features.csv --out work/audit_new
python -m chorus_hit.train_v2 --config configs/v2_quick.json --run-id v2_quick_002
python -m chorus_hit.train_v2 --config configs/v2_thorough.json --run-id v2_nested_002
```

The supplied `*_001` runs already exist and cannot be overwritten. Choose a new ID for a new run. To continue an incomplete run, repeat the same command with `--resume`; the code/config/data identities must match. Completed folds are retained. A supervised process group enforces the wall-time budget, and incomplete runs cannot serve predictions. Quick and nested configs have exactly the same 35 settings. Budgets are 1,200 / 3,600 seconds and two CPU workers. The initial observed runs took 16.39 / 77.18 seconds on this machine; other machines may take longer.

Both modes use only the original development membership. Ordinary thresholds remain fixed. Quick OOF scores are tuning evidence. The nested procedure includes candidate-family/representation selection inside every outer training fold. Different outer score scales make pooled nested ROC-AUC inappropriate; per-fold AUC is retained.

After freezing a completed development candidate, explicitly request the known benchmark:

```bash
python -m chorus_hit.evaluate_run --run-id v2_nested_002 --historical --out results/evaluations/v2_nested_002_historical
```

This creates a new, immutable evaluation folder. It does not select another candidate or promote the model. There is no automatic fresh-test command: a new collection needs a declared, locked protocol first.

## Prediction and software checks

```bash
python -m chorus_hit.predict --run-id v1_baseline --track-id CH0001
python -m chorus_hit.predict --run-id v2_nested_001 --track-id CH0001
python -m chorus_hit.predict --run-id v1_baseline --audio /path/to/your/song.wav --start 45
python -m pytest -q -m 'not integration'
python -m pytest -q -m integration
```

Use IDs in the original `results/test_predictions.csv` for historical examples. Other IDs may be training examples, and the CLI labels them accordingly. App/CLI/report loading verifies hashes, software versions, run identity, labels and the ordered raw input schema. Inputs with missing, extra, reordered or nonfinite features are rejected; there is no silent reordering. Upload extraction must match the bundle's extractor contract. CSV provenance hashes identify the stored dataset, not newly uploaded audio.

The unchanged six V1 tests are marked `baseline`; audio/app/process tests are marked `integration` from `tests/conftest.py`. The CI workflow runs the same full suite on Python 3.12 but has not been pushed or executed remotely.

## Safe V1 reproduction and exact executed V2 source

```bash
python -m chorus_hit.train --output-dir work/v1_reproduction_new
```

The destination must not already exist. The original V1 script was also run from a complete disposable archive before any fixes. `results/audit/v1_reproduction_comparison.json` records exact agreement. The full baseline source archive is `../Chorus_Hit_Baseline_V1.zip`.

`results/audit/v2_executed_code.zip` preserves the exact Python modules that produced both initial V2 runs, with its verified identity JSON. Later changes add reporting, optional audio interfaces and supervision without changing those measured results. Every run records the initial Git commit plus dirty state and executed source hash. Do not pretend that source hash is a clean commit.

## Reports and notebook

```bash
python scripts/build_reports.py --run-id v2_nested_001 --out work/reports_new --historical results/evaluations/v2_nested_001_historical --date 2026-10-02
python scripts/build_notebook.py --run-id v2_nested_001 --out work/Project_Walkthrough_new.ipynb
```

Choose new destinations to preserve existing exports. PDF generation checks one/two-page counts; review rendered output when content changes. Model name, feature count, uncertainty and conclusion come from the validated summary.

The editable presentation has a separate real builder, `scripts/build_presentation.mjs`; the Python PDF builder does not create PowerPoint. It consumes `docs/v2/presentation_input.json` and `learning_curve_chart.json`, generated from validated run data. Rebuild inside Codex's installed Presentation skill with its bundled `@oai/artifact-tool` runtime. Set `SKILL_DIR`, `RUNTIME_NODE`, `RUNTIME_NODE_MODULES`, `RUNTIME_PYTHON`, `RUNTIME_BIN_DIR`, `CHORUS_WORKSPACE` and `CHORUS_BUILD_DIR`; copy the builder into the private build directory and link its `node_modules` to the bundled modules. Run with the project root as the working directory. The builder finalizes to a new presentation path and refuses an existing final output. Change that destination for another revision. Review all rendered slides. No `python-pptx` or external PowerPoint installation is needed.

## Audio and embedding interfaces

```bash
python scripts/prepare_audio_dataset.py --help
python -m chorus_hit.embeddings --help
```

These interfaces are implemented and tested with software fixtures. Their real-data execution is blocked. Do not confuse a synthetic cache/extraction check with real-song accuracy. See `AUDIO_DATA_REQUIREMENTS.md` for the exact missing inputs and local-model review contract. There is no automatic recording or weight download.
