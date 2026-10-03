# Recording and data-source review — 2 October 2026

The user confirmed that they have no song recordings. The existing 751-song feature table remains the only matched input for the current year-end-hit versus other-chart-song task.

| Source inspected | What is available | Suitability for the current experiment |
| --- | --- | --- |
| [Original feature repository](https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus), pinned commit 838e76f | Collection/extraction notebooks and the feature table. A complete recursive tree contains 13 entries and no audio files. | Establishes feature provenance, but supplies no recordings for matched re-extraction or MERT. See `results/research_v3/upstream_audio_inventory.json`. |
| [ISMIR 2019 / Million Song Dataset hit-prediction release](https://zenodo.org/records/3258042) | Essentia audio features (18.6 GB archive), Billboard matches/non-matches, titles/artists/years. | Useful for a separately declared study, but the labels compare charted versus non-charted songs and the package does not list source waveforms. It cannot silently replace the current task or demonstrate improved chorus prediction. No large archive downloaded. |
| [Million Song Dataset](https://millionsongdataset.com/) | Publicly described as an audio-feature and metadata collection. Direct site retrieval timed out in this session. | No verified matching recordings obtained. The timeout is a retrieval limit, not proof that every related source is unavailable. |

No real audio or new labels were acquired. This was a focused source review, not an exhaustive claim that recordings cannot be obtained anywhere. No external dataset contributes to the V3 scores.

The usable next input is permitted recordings with recording/version identity, rights basis, verified chart labels and credited performers. The project already has a strict manifest and extraction interface for those files. A newly collected subset must report coverage by label and artist, rerun a matched baseline and declare its own grouping and evaluation protocol before comparing representations. See `AUDIO_DATA_REQUIREMENTS.md`.

Changing the label definition or adding post-release popularity measurements would change what the model is answering. Any such extension should have a separate task name and evaluation, rather than being reported as an improvement over the 52.9% chorus result.
