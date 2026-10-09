# New research data contract

For the current mini-project, start with the [simplified academic workflow](../../docs/hit_nonhit/academic_workflow.md) and `audio_intake.csv`. The strict canonical-entry importer below remains an alternative evidence policy.

**No real hit/non-hit dataset is bundled.** `provenance.json` deliberately records zero collected
candidates and no cutoff. Old `data/chorus_features.csv` labels must not be imported as new truth.

## Input files

1. `archive.json`: use `sources_manifest.schema.json`. Top-level chart/US region, snapshot ID,
   declared start/end, synthetic=false, source permission/license/reviewer and weekly issue objects.
   Each week contains date, evidence URL, reviewer, entries and canonical reviewed chart identities.
   `entries_sha256 = common.digest(entries)`; `source.export_sha256 = common.digest(weeks)`.
   These are hashes of normalized JSON payloads. Keep original export/license evidence privately too.
   Hashes are not authorization or proof that the provider is complete. Missing/partial weeks are
   accepted for auditing but never prove negatives. Wrong dates/ranks, duplicate identities and
   modified hashes fail closed.
2. `candidates.json`: list of records satisfying `schema.json`. Include independent canonical IDs,
   earliest release, album metadata, source/review evidence and synthetic=false. A pending/ambiguous
   candidate may omit unresolved IDs/dates; it stays in the unknown review queue without fabricated IDs. For training,
   `audio_path` must resolve within the supplied root; `audio_sha256` must match; processing permission,
   rights basis/reviewer/evidence and duplicate review must be supplied. An annotated chorus additionally
   needs reviewer/evidence, is_chorus=true and a timestamp. Do not invent review fields.
3. A **new local config copy** from `configs/hit_nonhit_v1.json`. After auditing authorized coverage,
   set the exact cutoff and record source/license/cohort decisions before data collection and fitting.
   Leave all predeclared split/search limits fixed; do not tune them after seeing a holdout score.

`synthetic_fixtures/` contains invented format examples. A single example issue is **not** complete
coverage. `python -m chorus_hit.hit_nonhit.smoke` generates its own complete synthetic chart and feature
fixtures. Random fixture features are not extracted musical evidence. Actual audio extraction is tested
separately with generated tones (including stereo/resampling and error cases).

## Implemented commands

Run with the repository's pinned Python 3.12 environment. Paths below are examples of future **user-supplied
local inputs**, not files claimed to exist. Each command implements `--help` and `--dry-run`.

```bash
python -m chorus_hit.hit_nonhit.chart_membership --archive output/hit_nonhit/inputs/archive.json --candidates output/hit_nonhit/inputs/candidates.json --config output/hit_nonhit/inputs/protocol.json --out output/hit_nonhit/v1/labels.json
python -m chorus_hit.hit_nonhit.build_dataset --archive output/hit_nonhit/inputs/archive.json --candidates output/hit_nonhit/inputs/candidates.json --config output/hit_nonhit/inputs/protocol.json --audio-root output/hit_nonhit/audio --out output/hit_nonhit/v1/dataset_001
python -m chorus_hit.hit_nonhit.freeze_split --dataset output/hit_nonhit/v1/dataset_001 --out output/hit_nonhit/v1/split_001
python -m chorus_hit.hit_nonhit.train --split output/hit_nonhit/v1/split_001 --out results/hit_nonhit/v1/run_001
python -m chorus_hit.hit_nonhit.evaluate_locked --split output/hit_nonhit/v1/split_001 --run results/hit_nonhit/v1/run_001
python -m chorus_hit.hit_nonhit.predict --run results/hit_nonhit/v1/run_001 --audio output/hit_nonhit/audio/authorized_song.wav
```

Add `--resume` only for an interrupted identical training run. Outputs are immutable once completed.
An interrupted locked evaluation consumes the test too; no second look. These local guards prevent
accidental reuse, not deliberate removal of a research ledger. Keep the original sealed split.

## Run the software check now without real data

```bash
python -m chorus_hit.hit_nonhit.smoke --out output/hit_nonhit/my_unique_smoke
python -m pytest tests/hit_nonhit -q
```

Choose a fresh output path. The smoke prints `synthetic: true`, `research_accuracy: null`, and
`real_data_training_status: blocked_data`; passing does not mean a genuine study succeeded.
No synthetic model is accepted in the app/prediction route.

## Publication boundary

Code, schemas, docs and invented fixture descriptions can be committed. Audio files, original/normalized
restricted chart exports, local identity evidence, full feature packages and trained research artifacts
stay ignored until their permissions permit publication. No secret or third-party recording belongs here.
This directory's schema/provenance placeholders do not grant any third-party rights.
