# Academic workflow: year-end hits versus non-charting songs

This is the **9 October 2026 revision**, requested for a two-week mini-project.
It supersedes the older “complete canonical identity for every weekly chart entry”
development requirement. The stricter importer remains available as a separate
explicit evidence policy. Faculty confirmation is a **submission checkpoint**.
There is no commercial archive subscription or formal permission-letter gate.

## What is reused

- All 751 original rows, 518-feature definitions, extraction/model infrastructure,
  and completed models/results remain unchanged.
- The 366 original class-1 songs are **candidate positives**, verified again using
  public year-end and weekly evidence. The initial exact-name audit verifies 344;
  22 remain pending alias/title review. MusicBrainz aliases may resolve additional
  candidates during album discovery.
- The 385 original class-0 songs are weekly-chart songs, explicitly excluded from
  the negative class. They are also excluded from this initial positive cohort.
- Existing audio features are **not copied into the new dataset**. Original audio
  and its exact extraction environment are unavailable. Both classes must be
  extracted together from permitted local recordings.

## Target and evidence limits

**1:** a reused original positive, corroborated in a 2006–2021 year-end Hot 100
ranking and the US weekly Hot 100 archive, with resolved recording metadata.
**0:** a comparable song absent from every complete weekly issue from its first
release through **30 December 2023** in the audited public archive. Both classes
need at least 24 months of follow-up. Weekly-only chart entries, uncertain matches,
missing weeks, imprecise dates, unresolved work identities, and conflicting
versions remain excluded/unknown. Chart entries preceding the claimed recording
release require review.

This is a song/artist-level historical comparison. Public chart credits do not
uniquely identify every audio master. The audio intake therefore separately asks
for confirmation of the named studio recording. “Verified” here means verified
against the **specified public metadata and rules**, not independently certified
by Billboard and not a claim of never charting anywhere or in the future.

The weekly snapshot passes continuity, 100 distinct ranks, and 100 distinct
normalized title/artist pairs for all **1,253 Saturday issues from 2000 through
2023**. These structural checks cannot prove every source entry is correct.
The provider documents compilation sources and known historical errors. The
historical observation window avoids its specifically documented 1961/1970 issues.

Sources, downloaded only as metadata, with URL/timestamp/SHA-256 caches:

- [UT Austin public weekly compilation](https://github.com/utdata/rwd-billboard-data),
  commit `24084586b74e6c4d40f95e90cf9a57f1b6388331`.
- [Public year-end compilation](https://github.com/Patrick5225/Billboard-Year-End-Top-Songs),
  commit `edb3d94c0517abbb023604c71ed9ca4549465965`, plus the
  [2021 year-end ranking, pinned revision 1362868665](https://en.wikipedia.org/wiki/Special:PermanentLink/1362868665).
- [MusicBrainz core metadata](https://musicbrainz.org/doc/About/Data_License),
  CC0, for artist aliases, album track lists, recording IDs, work IDs, and release dates.
  Its API is cached and rate limited to at most one request per 1.1 seconds.

Repository license files are preserved in the local source cache. They are not a
blanket audio license or authentication of Billboard's underlying data. Wikipedia
attribution/share-alike terms apply to relevant page material. The repository
publishes derived factual candidate metadata and source references; full snapshots
and **all audio** stay in ignored local directories.

## Candidate selection, before any modeling

The default bounded search selects the 40 original artists with most positive
candidates (name breaks ties), at most three dated studio albums each, prioritizing
albums near the positive candidates' year-end years. Releases are selected from
up to the first 100 official editions by MusicBrainz: earliest dated US edition,
else earliest dated available edition. This is a bounded convenience sample,
not an exhaustive discography. Live/DJ/remix/undated entries are not silently used.

Controls first match the same album, else the same canonical artist within two
release years. Discovery retains up to two possible controls per positive anchor.
The acquisition list pairs one unused control per positive and is capped at 300
songs. Each musical work and original positive is included at most once. A deterministic round-robin across artists retains artist variety. No
features, model predictions, or evaluation results influence this selection.
Incomplete API results and identity ambiguities are recorded, not filled in.

## Actual metadata result (10 October 2026)

The bounded discovery inspected 40 requested artists and retained 499 recording
candidates: 182 year-end positives, 263 songs absent under the public archive
query, and 54 unknowns (38 unresolved metadata identities, 11 chart/release-date
conflicts, five alternate mixes requiring review). Three negative candidates lack a comparable positive. These counts
are **metadata evidence**, not a completed audio-feature dataset.

The balanced acquisition list contains **278 songs: 139 per class**, from **36
artists**, all matched within the same albums. Its **34 connected metadata groups**
support the fixed split: 27 development groups / 7 held-out groups, and feasible
five-by-three nested validation. This is a preflight only: missing recordings and
actual audio duplicate checks may change support. No genuine test was frozen.

Two discovery issues are retained in the report: the `Ke$ha` artist search is
ambiguous, and one Travis Scott album candidate has no usable dated official
studio edition. Neither was guessed. The other artists/albums continued normally.

The published metadata is in `data/hit_nonhit_v1/academic/`. To use the supplied
candidate list, run only `prepare` below, then use
`--discovered data/hit_nonhit_v1/academic` with the intake command. Re-running the
MusicBrainz discovery is optional. In this local checkout the prepared archive is
already at `output/hit_nonhit/academic_prepared_final`.

## Commands

Use the pinned Python 3.12 environment from the project README. Run in the repository.
Every output name below must be new; source caches can be reused.

```bash
python -m chorus_hit.hit_nonhit.academic prepare \
  --cache output/hit_nonhit/public_cache \
  --out output/hit_nonhit/prepared
python -m chorus_hit.hit_nonhit.academic discover \
  --prepared output/hit_nonhit/prepared \
  --cache output/hit_nonhit/public_cache \
  --out output/hit_nonhit/discovered
```

The first command checks and reuses existing metadata. The second discovers
same-artist/albums, checks chart evidence, and writes `audio_intake.csv`, raw
candidates, labels/exclusions, source references, and a summary. It can take several
minutes because MusicBrainz requests are deliberately serial and cached. If a
request fails, its error remains in the report; rerunning to a new destination
reuses successful cache entries. No recording is downloaded.

### Simplest user input

Put permitted **WAV, FLAC, MP3 or OGG** recordings in one local folder. In a copy of
`data/hit_nonhit_v1/audio_intake.csv`, fill only these fields:

| Field | What to enter |
| --- | --- |
| `audio_path` | File path relative to your audio folder |
| `permission_basis` | Why you are allowed to process this recording |
| `permission_reference` | Source or permission reference; a concise explanation is sufficient, no formal letter required |
| `reviewer` | Your name |
| `recording_matches` | `yes` after checking the named studio recording/version |
| `chorus_start_seconds` | Start of a 15-second chorus, e.g. `42.5` |
| `chorus_confirmed` | `yes` after listening to that excerpt |

Leave unavailable recordings blank. Do not edit recording IDs or labels. Keep
both classes represented across many artists. Review all chorus positions where
possible; at least five per class are required before freezing a real split.
Do not copy audio into tracked repository folders or use ripped streaming audio.

```bash
python -m chorus_hit.hit_nonhit.intake \
  --prepared output/hit_nonhit/prepared \
  --discovered output/hit_nonhit/discovered \
  --intake path/to/filled_audio_intake.csv \
  --audio-root /absolute/path/to/permitted_audio --dry-run
```

After the missing-file/review list is resolved, remove `--dry-run` and add
`--out output/hit_nonhit/dataset`. This extracts **both** classes using the same
22,050 Hz mono, exactly 15-second, 518-feature pipeline. Automatic repetition
selection remains explicitly unverified when a chorus was not reviewed.

```bash
python -m chorus_hit.hit_nonhit.freeze_split \
  --dataset output/hit_nonhit/dataset --out output/hit_nonhit/split
python -m chorus_hit.hit_nonhit.train \
  --split output/hit_nonhit/split --out results/hit_nonhit/v1/academic_001
python -m chorus_hit.hit_nonhit.evaluate_locked \
  --split output/hit_nonhit/split --run results/hit_nonhit/v1/academic_001
```

The final command consumes the holdout **once**, even if interrupted. Do not open
`locked.csv`, delete the access ledger, search split seeds, or change the cohort
based on final-test performance. The Streamlit experimental section discovers a
compatible genuinely evaluated run automatically; the existing app model remains
available until then.

## Fixed comparison and leakage controls

`configs/hit_nonhit_academic.json` preregisters 27 candidates: dummy; logistic
regression with two regularization and class-weight settings; linear/RBF SVM;
random forest; extra trees; small regularized MLP; histogram gradient boosting.
Both all518/std74 are compared; two logistic/PCA95 variants compare learned
compression fitted **within each training fold**. MLP and boosting require at
least 15 development groups, otherwise they are explicitly skipped while the
simple models continue. No row-random early stopping is used.

Five outer and three inner group folds optimize balanced accuracy. Artist
(including guests), album, recording, work, match relationships and duplicate
signals form connected groups. The single seeded 20% final group holdout needs
at least five groups per class and remains unused during selection. A tiny or
highly connected cohort may require more artists, rather than weakening the split.
All transformations are fold-local. Timing is capped at 30 seconds per fit,
with an 18,000-second declared upper bound. Failed trials remain visible.

Development nuisance and label-permutation checks, class supports, precision,
recall, F1, confusion matrix, fit times, and group bootstrap intervals are retained.
The final locked test adds ROC-AUC and average precision from native scores.
These scores are not calibrated probabilities or future hit forecasts.

## What is still required

The repository contains **no permitted real song recordings for this new target**.
Metadata candidates are not a feature dataset. No new-target model has been trained,
no new holdout has been consumed, and no new accuracy is claimed. The old **54.61%**
accuracy remains historical evidence for a different classification target.
The 70–75% balanced-accuracy goal is an aspiration, not a promised result.
Faculty confirmation of the revised objective can be handled before submission;
it does not prevent implementation or data preparation.
