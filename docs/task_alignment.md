# Task alignment

approval_status: approved_as_reported_by_user
approval_recorded_date: 2026-10-05
approval_source: User response in this task: "Faculty has approved this adaptation"
independent_faculty_correspondence_available: false

The user confirmed faculty approval of the adaptation. This records that confirmation;
it does not claim that we received a separate faculty message or reproduced the paper exactly.

**Current task:** retrospectively distinguish the upstream Billboard year-end Hot 100
collection (1) from other sampled weekly Hot 100 songs (0). Both classes contain
charted songs. The user explicitly chose to keep this target.

**Reference:** Eric Liu's study uses charted versus other songs for its collected
artists. Our inherited 751-row feature table, label construction, normalized
artist-name split and selection protocol differ. Paper accuracy and runtime are
not matched experimental controls. The local MLP/forest benchmark is a comparison
on our data, not a reimplementation of Eric's entire collection/extraction study.

**Deliverable:** reproducible precomputed-feature study, preserved baseline,
selectable research models and experimental uploads. A greater-than-75% score is
a stretch goal, not a numerical requirement found in the supplied assignment guidelines.
The separately numbered problem statement is still unavailable; the approved
adaptation above defines this implementation's scope.

**Later handoff:** the repository remains public at the user's instruction. The
assignment's private-submission requirement is a later user/faculty handoff item.
The final PDF write-up and slides are supplied separately from the code repository.

## Current work scope — 2026-10-06

The final bounded accuracy check is complete. A fixed equal-weight ensemble of
the eight existing Chorus Hit Predictor std74 logistic settings scored 55.28% accuracy, versus
56.62% for the fixed Chorus Hit Predictor control on identical development folds. No improvement
was found. Accuracy experimentation is stopped, and the active baseline is
preserved. These exploratory scores do not replace the nested Chorus Hit Predictor assessment or
provide a fresh-test accuracy claim.

Rehearsal, Q&A preparation, and further faculty/access confirmation are **deferred
by the user**, outside the current work list. Faculty adaptation approval above
remains recorded; no additional confirmation is requested. The repository must
remain public.

## Final submission materials — 2026-10-07

The user subsequently requested the final PDF and presentation, lifting their
earlier deferral. The two-page write-up and 12-slide deck use the saved Chorus Hit Predictor nested
result (54.61% accuracy), distinguish the exploratory fixed-control score
(56.62%) and rejected ensemble (55.28%), and retain the data limitations. They
remain outside GitHub to keep the repository focused on code and evidence.
The tested fixes were merged into main through PR #5. Rehearsal, Q&A and further
faculty/access confirmation remain deferred; no new accuracy search was run.
