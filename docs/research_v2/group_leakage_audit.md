# Group, source and leakage audit — development only

Existing identity graph validation confirms disjoint known connected groups
across frozen partitions. The new research reads only development labels;
metadata identifiers are used for grouping/diagnostics, never as predictors.
Unknown artist aliases, covers, alternate masters and metadata omissions remain
unverifiable without independent review; passing the graph check does not prove
they are absent. The original held-out identity coverage is 527 groups,
from the immutable split receipt; no new held-out error analysis was performed.

## Group and feature checks

- Development: 2,106 groups / 6,235 rows; size quantiles `{'0.0': 1.0, '0.5': 1.0, '0.9': 6.0, '0.99': 17.0, '1.0': 636.0}`.
- Pure-label groups: 1694; mixed-label groups: 412.
- Duplicate exact development feature vectors: 0.
- Missing cells: 0; constant features: 34.
- Descriptive feature pairs with absolute Pearson correlation >.98: 222.
  This was not used for post-hoc feature removal.
- Groups with at least one nested OOF error: 875.
  Group-accuracy quantiles: `{'0': 0.0, '0.25': 0.5, '0.5': 1.0, '0.75': 1.0, '1': 1.0}`. Many small
  groups have discrete 0/1 accuracy, so these are descriptive, not reliable
  individual-artist performance estimates.

| Class | Rows | Groups | Missing cells | Median duration (s) | Duration p10 | Duration p90 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3084 | 1517 | 0 | 231.09 | 156.16 | 342.62 |
| 1 | 3151 | 1001 | 0 | 212.86 | 146.26 | 294.95 |

## Codec and source distribution

| Codec | Rows | Class 0 | Class 1 | Groups | Nested OOF accuracy |
| --- | --- | --- | --- | --- | --- |
| aac | 395 | 107 | 288 | 219 | 72.66% |
| alac | 5 | 1 | 4 | 5 | 100.00% |
| flac | 1046 | 747 | 299 | 489 | 75.62% |
| mp3 | 4466 | 2053 | 2413 | 1646 | 70.11% |
| mpc7 | 1 | 1 | 0 | 1 | 100.00% |
| vorbis | 311 | 168 | 143 | 173 | 79.42% |
| wavpack | 1 | 1 | 0 | 1 | 100.00% |
| wmav2 | 10 | 6 | 4 | 8 | 70.00% |

Per-codec scores are post-run descriptive slices, not separately selected models;
very small codecs cannot support reliable estimates. All development rows report
extractor `{'2.1-beta2': 6235}`. [Sample-rate counts and descriptive metadata](descriptive_diagnostics.json)
are also published. No genre or reliable release-era fields exist in the frozen
metadata; no guessed positional join or label-derived era was substituted.

Predeclared baseline controls, each with 3-fold grouped OOF evaluation:

| Subset | Rows | Groups | Accuracy | BA | BA 95% group interval |
| --- | --- | --- | --- | --- | --- |
| MP3 codec only | 4466 | 1646 | 69.12% | 68.79% | 67.15%–70.69% |
| Without largest connected group | 5599 | 2105 | 71.48% | 71.57% | 70.06%–73.06% |

These restrict codec / remove the single metadata-defined largest component;
they do **not** create a fully matched collection. Fold/cohort composition differs
from the full 5-fold experiment. Samples were not removed based on prediction
errors and these controls did not determine the deployed model. A causal source-
invariant representation and genre/era-matched benchmark remain unavailable.

## Permutation interpretation

Global shuffled-label BA (three predeclared permutations):
**50.17%, 47.58%, 49.17%**.
These destroy overall descriptor/label association while retaining the group
splits. They are sanity controls, not calibrated exchangeable-group hypothesis
tests or a p-value. The original within-group permutations retain group label
structure, since pure-label groups cannot change. Their ~66% scores therefore
do not contradict global-null behavior or establish a software bug. They warn
that class-pure identities and cohort/source structure matter.

## Feature-family association (not causation)

Across three grouped validation folds, mean BA drops after jointly permuting
one family across validation rows:

| Family | Mean BA drop (percentage points) |
| --- | --- |
| lowlevel. | 15.42 |
| rhythm. | 2.32 |
| tonal. | 2.03 |

This perturbation can break correlations between families. It is model
sensitivity, not causal songwriting advice, a calibrated probability, or an
individual-song explanation. See [raw diagnostics](results.json).
