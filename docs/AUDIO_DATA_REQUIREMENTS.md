# Inputs needed for the blocked audio study

No real-song embedding scores have been measured. `data/recording_audit_v1.json` lists each of the 751 existing track IDs with missing fields. `results/audit/audio_coverage_by_label_artist.csv` shows zero eligible recordings for both labels and every artist.

For each new recording, provide:

- A permitted local WAV/FLAC recording or authorized reference, the rights basis and a SHA-256 checksum. No recording rights follow automatically from the feature-table license.
- Recording identity/version (album, single, remix, clean/explicit), full credited performer identifiers, evidence sources and duplicate links.
- A verified year-end/other-chart label under the same declared US chart coverage and observation window. A weekly chart entry alone does not verify the negative class.
- Release date and chart evidence dates; a forecasting variant additionally needs a prediction cutoff and completed outcome window.
- One shared extraction version and chorus-selection policy applied to both classes. Current interface accepts one 15-second segment per recording; multi-segment aggregation is not implemented.

The manifest schema and validator are in `chorus_hit/ingestion.py`. After completing the evidence, `scripts/prepare_audio_dataset.py --manifest FILE --out NEW_DIRECTORY` validates and extracts features. This interface is tested with synthetic fixtures only. It has not produced a real audio dataset. The comparable CSV runner deliberately rejects a new dataset/version; a new study must declare its own grouped folds and baseline before training.

For MERT, also supply a local copy of revision `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`, reviewed hashes of every code/config/weight file, compatible optional torch/transformers dependencies and a license-compatible intended use. The official processor uses 24,000 Hz. The official repository lists 378 MB PyTorch weights; downloading the 1.32 GB fairseq checkpoint is unnecessary for this adapter. No optional dependencies or model weights have been installed/downloaded. CPU mode is implemented but its real-model speed/memory use is unmeasured.

`configs/mert_optional.json` records the revision, processor, last-layer mean pooling and license. The adapter refuses to load unreviewed local custom code and makes no network download. Its code/weight review is still pending. [Official model card](https://huggingface.co/m-a-p/MERT-v1-95M), [pinned revision](https://huggingface.co/m-a-p/MERT-v1-95M/commit/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5).

A genuine fresh test collection is also absent. Lock its inclusion rules, performers/duplicates, time coverage and labels before selecting a final model. Repartitioning these 751 already examined rows cannot create fresh confirmation. No specific collection size guarantees the 75% target.
