# Development-only research protocol

Fixed before any new fit: seed 20261008; original 6,235 development songs and
artist/recording-connected groups; five outer / three inner folds; eleven candidates
in `configs/research_v2.json`. Select inner mean balanced accuracy, break ties by ID.
Quantile/robust scaling and supervised selection fit only on each training partition.
Compare fixed RBF baseline, larger C/gamma, feature selection, boosting, extra trees,
and a hard-voting ensemble. No fitted stacking or ungrouped early stopping.

Maximum 90 seconds per fit; 7,200 accumulated trial seconds. Record each trial,
warnings, failures and elapsed time. A failed candidate is ineligible for selection
in that fold. Outer-fit failure makes the study incomplete. No seed retries.

The old test and its ledger are immutable. The new script loads only the verified
development Parquet. No test function is called. Selection and diagnostics are
exploratory development evidence; there is no new independently untouched cohort.
The deployed model is not changed even if a candidate scores better.

Predeclared descriptive controls: codec/extractor/group/class tables; missingness
and duplicates; three global-label permutations under a no-association null;
three-fold baseline on the MP3-only subset and without the largest identity group;
three-fold feature-family permutation importance. These controls cannot trigger
cohort or hyperparameter revisions. Global permutations are not a within-artist
exchangeability test. Family importance measures model associations, not causation.
Release era and genre will be reported unavailable if absent from frozen metadata.
No source-wide scan of held-out records will be used to invent these fields.
