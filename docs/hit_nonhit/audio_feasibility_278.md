# Audio acquisition feasibility — 278-song cohort

Checked 10 October 2026. **BLOCKED_AUDIO** for the requested repeated-chorus experiment.

The bounded investigation found no source establishing both download permission and
local analysis permission for a usable matched cohort of the selected studio recordings.
This is a finding about the sources checked, not a claim that no permitted copy exists anywhere.
The user confirmed that they have no recordings. No commercial audio was downloaded.
Investigation started with the local audit at 03:54 UTC; the acquisition decision was
recorded at 04:36 UTC (41.25 minutes after the audit began), within the three-hour cap. Subsequent work is implementation,
verification and reporting, not continued open-ended acquisition searching.

## Actual coverage

| Item | Hit | Non-charting candidate | Total |
| --- | ---: | ---: | ---: |
| Fixed metadata candidates | 139 | 139 | 278 |
| Permitted exact recordings obtained | 0 | 0 | 0 |
| Reviewed exact recordings | 0 | 0 | 0 |
| Human-reviewed choruses | 0 | 0 | 0 |
| Compatible 15-second/518-feature rows | 0 | 0 | 0 |
| AcousticBrainz recording IDs with feature submissions | 129 | 134 | 263 |
| AcousticBrainz IDs without submissions | 10 | 5 | 15 |

Audio acquisition completion: **0/278 (0%)**. Both-class audio match sets: **0**.
Actual audio groups: **0**. Missing recordings: **278**. Rejected files, wrong-version
files and decoder failures: **0**, because no files were received or downloaded.
These zeros are inventory counts; they are not model performance measurements.

The fixed list has 36 lead artists, 67 albums and 34 connected **metadata** groups.
Those groups do not prove that a usable audio train/test split exists.

## Sources and automated acquisition routes

| Source / route checked | Evidence and outcome |
| --- | --- |
| [Spotify Developer Policy](https://developer.spotify.com/policy) | Sections III.13–14 prohibit the analysis/ML use at issue. A plugin, Premium subscription or accessible preview URL does not establish authorization. No audio requested. |
| [Apple Search API terms](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/) | Preview permission is promotional streaming; the published terms prohibit downloading/saving/caching previews. Rejected as an acquisition route. Previews also do not establish chorus coverage. |
| [FMA](https://github.com/mdeff/fma) / [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) | Documented bulk music-research datasets. Their descriptions do not establish exact matches for this mainstream cohort or our Billboard labels. No substitute songs were imported. No exhaustive catalog match is claimed. |
| Exact-title/version searches | Bounded searches included Adele / Hello / 25 / recording `0a8e8d55-4b83-4f8a-9732-fbb5ded9f344` and Adele / I Miss You / 25. Results included metadata and lyrics; none established permitted analysis of the exact audio. A CC license on a wiki or translated lyrics does not license the recording. This was a representative audio-source search, not 278 individual rights reviews. |
| [Eric Liu reference paper](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf) and exact-title repository searches | No downloadable, rights-documented feature release satisfying our exact identities, feature order, normalization and shared extractor contract was located. A paper's downloader method does not confer present permission. |
| Hugging Face | Connector returned `MCP tool dataset_search was not returned by tools/list`. Public Hub API searches succeeded for `hit song`, `hit-predictor`, `chorus`, and `billboard`; results did not establish a compatible licensed 278-song chorus table. No account/token purchase is needed for the alternatives below. Search failure/limited results are not proof of universal absence. |
| [AcousticBrainz](https://acousticbrainz.org/) | Working public bulk API; exact-ID lookup completed for all 278 in 12 requests, at most 25 IDs per call. 263 have submissions. Data is CC0. This source stores features, **not audio**. |

No paid catalog was purchased, no rightsholder contacted, and no streaming restriction
was bypassed. Bulk downloads would become useful only after source-specific permission
and exact version matching are established. No permissive-looking URL was promoted
automatically to an authorized recording.

## Exact-list feature alternative: AcousticBrainz

The batch count response covers 263 IDs (94.6% of the list); there were no failed
requests or merged-ID mappings in this run. Their metadata spans 35 lead artists and
65 albums. All 134 available controls retain positive metadata support under the
existing matching rule. This is **availability**, not 263 validated training rows.

One actual low-level document, for Adele's Hello, was inspected: its analysis sample
rate is 44,100 Hz and analyzed duration about 295.1 seconds. It uses Essentia, not our
pinned 22,050-Hz/15-second librosa extractor. It cannot be concatenated with the old
751 rows or relabelled as repeated-chorus features. See the provider's
[data description](https://acousticbrainz.org/data) and [no-audio FAQ](https://acousticbrainz.org/faq).

A separate whole-recording experiment could retain much of the selected cohort.
It would first need deterministic submission/version selection, identity and extractor
consistency checks, audio-only numeric columns, source/codec diagnostics, and a fresh
preregistered group evaluation. No such model was trained or selected here.

## Recommended published Hit vs Non-Hit fallback: HSP-S

Among the sources checked, the best ready-labelled research candidate is
[HSP-S, by Vötter, Mayerl, Specht and Zangerle](https://zenodo.org/records/5383858)
(DOI `10.5281/zenodo.5383858`). The record API declares **CC BY 4.0**.
The [authors' paper](https://dbis-informatik.uibk.ac.at/sites/default/files/2022-06/ISM_2021__Hit_Song_Prediction.pdf)
describes Billboard-matched hits and sampled MSD non-hits. Those are the authors'
labels, not our independently audited absence-through-2023 labels.

Live file inspection, beyond the README:

- `hsp-s_acousticbrainz.parquet`: footer reports **7,736 rows, 1,890 columns**.
- Chart columns fetched with bounded HTTP range requests: **3,870** rows with
  positive `peakPos`/`weeks`, **3,866** without chart entries. The published description
  rounds this to a balanced 50/50 dataset.
- `hsp-s_uuid_year.csv`: **44,272 unique UUID rows**, despite the description's
  7,449 release-year figure. Its MD5 matches the publisher (`572a6a94ac783459ec141e89a834aed3`).
  This mapping needs a validated identity join; never attach years by row position.
- Full feature download timed out after a partial transfer. Footer/label ranges
  succeeded. The complete file MD5 and all feature rows are **not yet verified**;
  the partial download is not a usable dataset or a completed experiment.

For a separate approved HSP-S experiment, use one coherent audio feature source.
Exclude chart rank/weeks, listener/play counts, filenames, tags and identifiers from
predictors; keep identity fields only for deduplication and artist grouping. Audit
negative labels, missingness and group support before freezing any test. These
features are not our 518 chorus features, and would not directly power the present
librosa audio-upload route without a separately compatible inference extractor.

The common Spotify Hit Predictor benchmark is a less suitable default here: its
negative-sampling provenance and Spotify-derived data rights need their own review.
No accuracy comparisons across these differently defined tasks are justified.

## Local ledger and reproducibility

The acquisition command creates a fresh **ignored** ledger and editable intake for
all 278 IDs, with the required source, rights, version, path, hash and chorus fields.
All missing evidence remains blank/unknown; all rows begin `MISSING`. The ledger is
an acquisition worksheet, not an authorization certificate.

```bash
python -m chorus_hit.hit_nonhit.acquisition init \
  --out output/hit_nonhit/real_audio_48h_20261010/intake
python -m chorus_hit.hit_nonhit.acquisition probe-features \
  --out output/hit_nonhit/real_audio_48h_20261010/feature_probe
```

Use a new output name when repeating; existing results are never overwritten.
The probe retrieves **counts only**, distinguishes unavailable/failed queries from
missing submissions, respects rate limits, and downloads no recordings.

## Decision gate

**Recommended next action: approve HSP-S as a separate feature-based experiment.**
This is a scope decision required by the supplied plan, not an account-access request.
The 278-song repeated-chorus project, historical model and dataset remain intact.

To resume the exact-278 audio route instead, the missing input is a source or provider
grant covering download and local analysis of the selected recordings, with recording
identifiers/versions. It can be a permitted batch delivery; individual manual MP3
downloads are not required. After acquisition, a real person must check recording
identity and at least five chorus excerpts per class before the fixed split gate.
