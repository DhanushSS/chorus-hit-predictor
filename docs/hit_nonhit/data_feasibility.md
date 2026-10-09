# Data feasibility — 2026-10-08

> Historical October 8 status/strict-import policy. The October 9 academic revision
> uses public metadata and a fixed cutoff; see [current workflow](academic_workflow.md).
> Faculty confirmation is a submission checkpoint. Preserved results below are historical.

**real_data_training_status = blocked_data**

| Dependency | Observed availability | Permission/coverage status | Consequence |
|---|---|---|---|
| Original feature CSV | 751 inherited rows; preserved | Provenance exists for the old collection; zero independently verified new-task labels | Cannot relabel or mix with new audio |
| Complete US weekly Hot 100 archive | No authorized export supplied/found | Official chart-page browser retrieval failed; no historical completeness or reuse permission verified | 0 verified weeks; negative labels and cutoff blocked |
| Canonical recording/artist/work/album metadata | MusicBrainz documentation accessible via search | Core metadata has CC0 licensing; supplemental/documentation terms differ; no specific cohort reconciled | Metadata source is a lead, not verified recording identity or audio rights |
| Both classes' lawful audio | No recordings/authorization supplied | No acquisition or redistribution rights established | 0 usable real recordings/features |
| Human chorus/duplicate review | No permitted clips supplied | 0 reviews, accuracy unavailable | Real segmentation and duplicate audit blocked |
| New target faculty approval | Only old adaptation was previously approved | Pending separate confirmation | Academic alignment still requires confirmation |

The original [Stanford paper](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf)
uses discography controls and a 15-second chorus, but its sampling and evaluation differ. Its acquisition
method does not grant us recording rights. No audio ripping or streaming downloads were performed.

Source leads checked: [official Hot 100 page](https://www.billboard.com/charts/hot-100/),
[MusicBrainz data licensing](https://musicbrainz.org/doc/About/Data_License),
[MusicBrainz core-data downloads](https://musicbrainz.org/doc/MusicBrainz_Database/Download), and
[MusicBrainz API](https://musicbrainz.org/doc/MusicBrainz_API). A publicly visible chart or a subscription
is not evidence that a complete research export is available or licensed for redistribution.
The current failure to establish access is not a claim that no legitimate source exists.

## Implemented acquisition boundary

The adapter reads authorized **local** normalized exports. Each source needs provider, permission basis,
license snapshot, reviewer/date and payload SHA-256. Each issue needs date, evidence URL, reviewer,
entries hash and distinct ranks. Exact recording IDs require explicit evidence and version review.
Missing issues/ranks remain visible; they do not silently disappear from the denominator. A positive can
use direct evidence; a negative additionally needs the complete window and reconciled entries.

Permission fields are human assertions with provenance, not automatic legal verification or authenticated
provider signatures. Do not set them true merely to pass validation. Store original exports and license
snapshots privately, then preserve normalized payload hashes. No paid source, account or subscription
has been opened. No real chart export or recording is committed.

## Counts and estimate

Real collected candidates: **0**. Verified positives: **0**. Verified negatives: **0**. Unknown candidates:
**0** (no candidate cohort has been imported). Real artists, albums and match sets: **0**. Coverage weeks:
**0**. Reviewed real chorus clips: **0**. There is no defensible estimate of final cohort size yet.
The synthetic fixture contains deliberately invented records and charts and is never a cohort estimate.

A pilot must first supply enough independent connected artist groups for class support. Matching may
connect many artists through collaborations, so song count alone does not establish feasibility. For the
primary configuration, both classes must support the locked holdout plus five outer / three inner folds.
No real-record provenance walkthrough can be written until an authorized record exists.
