# Audio and evidence boundaries

`schema_validated: true`; `waveform_parity_verified: false`.

`extract_record` records these statuses, the audio SHA-256, extractor configuration,
implementation hash, segment start/duration, selection method and repetition similarity.
518 features in the exact `FEATURE_COLUMNS` order are required. Raw uploads and
precomputed data must be finite. NaN and infinity are rejected at data ingestion and
prediction; the saved imputer remains in existing pipelines for byte compatibility.
Invalid data is rejected before fit/save/reload rather than silently introducing a
new missing-value protocol.

Audio is decoded to mono float32 at 22,050 Hz, without amplitude normalization,
then a 15-second sample-bounded segment is selected. A manual time is rounded to
the nearest sample. Negative, nonfinite or out-of-bounds times fail. Clips under
15 seconds, silence and clips over eight minutes fail.

Each feature channel uses skew, min, max, population standard deviation, mean,
median and Fisher kurtosis, in that order. Near-constant skew/kurtosis are zero.
STFT-based features use 512-sample hops and reflect padding; the explicit frame
size is 2048 where applicable. Remaining CQT/tonnetz details follow the pinned
librosa defaults recorded by `extraction_config()`. It is not claimed that all
feature families use the same transform or frame size.

`legacy-librosa-518-v1` deliberately reproduces the upstream positional
zero-crossing-rate call: frame length equals the sample rate (22,050).
`shared-librosa-518-v2` uses 2048. These contracts are incompatible; neither the
legacy feature table nor its saved models was silently migrated.

The automatic selector is a repeated-segment candidate based on chroma similarity,
not pychorus and not verified chorus detection. It can select a verse. Short clips,
few candidates or no separated repeat use a labelled centered excerpt. Manual,
automatic and fallback selections are retained in metadata and displayed in the app.

## What remains unmeasured

Original source waveforms and reference feature values are unavailable. Synthetic
tones verify software/decoding, not waveform equivalence or new-song accuracy.
No genuine song has a recorded chorus annotation review in this task (denominator 0).
A future permitted review must declare timestamp tolerance and include failures and
songs without conventional choruses. Its tolerance and accuracy remain null until
that protocol exists; do not invent results.

New recordings require evidence URLs, chart market/type/window, canonical IDs,
recording version, rights basis, file hash, compatible extractor and valid boundaries.
Structural checks against supplied references do not independently authenticate
those references. `prepare_audio_dataset` prepares features; it does not make the
legacy trainer compatible with connected canonical identity groups.
