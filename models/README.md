# Evaluated HSP-S model

`hsp_s_run_001.zip` contains the original, immutable run: trained model, feature
schema, configuration, validation checkpoints, diagnostics, predictions and the
one-use test result. Keeping one compressed bundle avoids hundreds of generated
files in the repository browser.

Run `python scripts/setup_model.py` from the project root. The installer checks
the archive SHA-256 in `bundle.json`, rejects unsafe archive entries, verifies the
original per-file manifest, code identity and dependency versions, then installs
to ignored `results/hsp_s/run_001/`. It never replaces an existing differing run.
Only the project's trusted bundle should be loaded; hashes do not make arbitrary
pickle/joblib files safe.

The model was trained once on 6,235 HSP-S development records and evaluated once
on 1,493 held-out records. Packaging does not retrain, retune or re-evaluate it.
Dataset: Vötter, Mayerl, Specht and Zangerle (2021), DOI 10.5281/zenodo.5383858,
CC BY 4.0. Feature subset, preprocessing and evaluation changes are documented in
`docs/hit_nonhit/hsp_s_protocol.md` and the result reports.
