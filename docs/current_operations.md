# Current application operations — 11 October 2026

The active entry point is `app.py`. Run `python scripts/setup_model.py`, then
`python -m streamlit run app.py`. There is no `pages/` entry point in this version.
The old handoff is a historical research snapshot; its original app layout and
chorus-upload instructions do not describe the current checkout.

The packaged RBF SVM remains the evaluated model. New development experiments
have no automatic promotion path. CSV classification and MP3 descriptive preview
are separate modes. See the [upload contract](audio_upload/upload_contract.md).

`chorus_hit/config.py` keeps historical constants because retained shared modules
import them. Current HSP-S inference uses its bundle's own 436-column schema and
`results/hsp_s/run_001`; old model/data constants are not current instructions.
No hashed HSP-S implementation was edited, so the model identity remains valid.

## Loading performance

[Five local measurements](audit/load_benchmark.json), during concurrent research:
median full verified load 0.113 s; median four-row prediction including validation
0.132 s. The measured process RSS includes imports and prior work, not isolated
model memory. This does not establish excessive loading overhead. No cache was
introduced: every rerun retains full manifest/code/package integrity checks,
including after artifact tampering. Reconsider caching only if deployment latency
warrants a fully tested invalidation scheme.

## Research and archive

Historical model/report/split/one-use ledger remain intact. Development research
is in `chorus_hit/research_v2/`; do not run the old test evaluator to obtain a new
score. See [migration](repository_migration.md) and the public
[archive](https://github.com/DhanushSS/chorus-hit-predictor-archive-2026-10-11).
