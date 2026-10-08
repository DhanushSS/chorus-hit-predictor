# Chorus Hit Predictor — new research task

**Model availability: none.** Task `hit_nonhit_v1`; training status `blocked_data`.
The application's existing model remains active with its original year-end/other-charted meanings.

Intended input: 15 seconds of authorized audio, 22,050 Hz mono, homogeneous `shared-librosa-518-v2`,
518 ordered measurements. Intended output: retrospective US weekly Hot 100 membership by a fixed
verified cutoff, versus complete-window nonmembership. Cutoff: pending. Unknown ground truth is
excluded; it is not a third prediction class.

Training cohort, sizes, licensed sources, real feature-extraction failures, selection accuracy, nested
metrics, locked metrics, CI, per-artist performance and measured real inference timing: unavailable.
The implemented predeclared search includes dummy, logistic, linear/RBF SVM, random forest and extra
trees. Conditional boosting/neural models and representations require adequate independent evidence.

Out of scope: predicting future fame, quality, streaming success, never charting anywhere, or calibrated
commercial-success probabilities. Matched-control prevalence does not represent all music. Audio codec,
mastering and source differences may confound results. Human chorus and duplicate checks remain pending.

Deployment: new target is opt-in only after a genuine compatible bundle and locked evaluation exist.
Synthetic models and legacy bundles are rejected by the new task loader. No new default or accuracy
claim accompanies this implementation. Packages are pinned; unsupported training platforms fail closed.
