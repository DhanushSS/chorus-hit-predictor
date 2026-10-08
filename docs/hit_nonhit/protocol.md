# Charted versus verified non-charted protocol

**Task:** `hit_nonhit_v1`. **Label rule:** `us_hot100_release_to_cutoff_v1`.
Research question: can 15-second chorus audio distinguish US weekly Billboard Hot 100
charted recordings from comparable recordings absent from every eligible chart issue?
This is retrospective; future success and population-wide hit prevalence are outside scope.

## Window and identity

The exact cutoff is **pending, null in configuration**. No complete authorized archive is available.
The code refuses real labeling without an explicit Saturday issue-date cutoff and refuses feature
building unless that cutoff issue has all 100 distinct ranks. Choose and commit the cutoff only
after a licensed/authorized archive is audited, before sampling and fitting. A historical archive
with different publication weekdays needs a reviewed schedule adapter/new protocol, not guessed weeks.

All recordings require at least 24 calendar months from the earliest public release to cutoff.
An earlier single or promotional release takes priority over album release. Unknown release precision,
ambiguous credits, aliases, translations, remixes, live recordings, clean edits or chart grouping
require canonical-ID review. String normalization is only a review aid. A chart entry with a related
work but no reviewed matching recording cannot support a negative. Chart appearances predating the
claimed first release require correction and review. Re-entry before cutoff makes a recording positive;
a first appearance after cutoff cannot change the frozen label.

- **1:** canonical recording has reviewed US weekly Hot 100 evidence in the release-to-cutoff window.
  Positive evidence can exist even if some other chart weeks are missing.
- **0:** every Saturday issue in that entire window has all ranks 1–100 and reviewed chart identities,
  and the canonical-recording/work query returns no appearance or unresolved related version.
- **Unknown:** incomplete coverage, unresolved identities/versions, conflicts or insufficient follow-up.
  Never included in fitting/evaluation. Rights/audio failures preserve the separate label audit but
  exclude that recording from audio modeling.

A fixed 24-month *outcome* window is not a substitute: 24 months here is minimum follow-up, and the
absence query runs all the way to the fixed cutoff. There is no claim a negative will never chart.

## Sampling, audio and review

Keep every eligible control with a same-artist/same-album positive; fallback to a shared canonical
artist within two release years. Log level, positive IDs, viable matches and year distance. Do not
subsample based on scores. No metadata enters audio estimators. Artist, album, work and match IDs
are used only for sampling, audits and leakage prevention. The resulting class prior is artificial.

Both classes use `shared-librosa-518-v2`: 22,050 Hz mono, exactly 330,750 samples, fixed 518-feature order.
Hash decoded-file inputs, extractor settings/code and feature rows. Supported decoded types: WAV,
FLAC, OGG and MP3 as supported by pinned libsndfile. Reject wrong extension, silence, invalid samples,
under-15-second or over-480-second recordings and out-of-bounds segments.

Methods are `manual_reviewed`, `repetition_heuristic`, or `fallback_excerpt`. A chosen timestamp alone
is not reviewed chorus evidence. Minimum real split gate: five reviewed clips per class; reviewer,
annotation evidence, chorus judgment and start within 0.5 seconds must be recorded. This minimum
is a quality gate, not enough to establish detector accuracy. Currently reviewed real clips: **0**.
A conservative whole-recording chroma signature groups highly similar candidates (cosine >=0.995);
independent duplicate/version review and any robust fingerprint cluster IDs are also required.
This heuristic does not prove all perceptual duplicates have been found.

## Freeze and model selection

Connected components join every shared artist (including guests), canonical work/recording, album,
match-set/positive link, exact audio/feature hash and reviewed or heuristic duplicate cluster.
Seed **20261008**, one GroupShuffleSplit with 20% of groups reserved, no seed search. Reject missing
classes or inadequate groups. Require >=5 independent groups supporting each class in both partitions,
plus feasible five outer / three inner development folds. Group isolation can make a small cohort unusable;
that is a blocker. No secondary within-artist score replaces this primary unseen-artist claim.

Freeze schema, data/config hashes and membership before selection. Development and locked CSVs are
separate. The trainer hashes but never parses the locked table. File permissions are not a cryptographic
blind: researchers must also refrain from manually opening it. The source data was necessarily seen
at collection, so this is a procedural test-label seal, not inaccessible truth.

The configuration fixes **21 candidates**: dummy prior, regularized logistic regression with both
class-weight settings, linear/RBF SVMs, random forest and extra trees on all518/std74 representations.
Rank by mean inner-fold balanced accuracy, deterministic candidate-ID ties. All variance filtering,
scaling and optional PCA live inside fitted pipelines. No global imputation or resampling. Native
thresholds only; no test-dependent calibration. Conditional MLP/histogram boosting needs >=50 development
groups and a newly preregistered config before holdout access; embeddings and new selectors are deferred.

Fits use one numerical thread, <=30 seconds per fit and a 14,400-second total upper bound checked from
the predeclared fit count. Checkpoints record membership, warnings, failures and fit times. Resume only
identical data/config/code/environment. Completed outputs cannot be replaced. A crashed process may
leave `.training.lock`; inspect the process and preserve checkpoints before manually removing a stale lock.

Three within-group label permutations and a development-only nuisance classifier examine source, codec,
duration, release year and sample quality. Nuisance BA >=0.65, suspicious permutation performance, or no
within-group exchangeability blocks real locked evaluation; review and redesign the cohort before touching
test results. Every candidate and failure is retained, not just the winner.

## One final test and reporting

Fit the frozen chosen pipeline on all development records. The split owns an exclusive `test_access.json`;
all runs share that access gate. Write it **before** reading labels. Even interruption consumes access;
restarting is not an excuse to repeatedly test. This prevents accidental repeats, not deliberate deletion
by a filesystem owner. Never delete the ledger or regenerate the split after seeing outcomes.

Report nested development separately from locked test: BA (primary), raw accuracy, macro F1, positive
precision/recall, negative recall, confusion matrix, class/prediction support, native-score ROC-AUC and
average precision when meaningful, per-group support and 95% connected-group bootstrap CI. CI is
unavailable with fewer than five independent groups per class. Extraction time and inference time are
separate. A high number without audited inputs is not acceptable evidence.

Faculty approval of this **new** target: pending. Independent software work proceeds as authorized.
