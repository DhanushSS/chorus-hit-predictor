# Upload contract

## CSV classification — working

- UTF-8 (BOM accepted), comma-separated, exact 436 unique scalar names from the
  downloaded template. 1–100 data rows; maximum 5 MiB at widget **and** parser.
- Matching names may be reordered only through the explicit checkbox. Missing,
  extra or duplicate columns are rejected; no metadata/chart field is a predictor.
- Numeric cells only, maximum 128 characters per cell; infinity and all-missing
  rows are rejected. Blank/NaN cells use the already-trained median imputer.
- Legacy 518-feature files receive a specific explanation, never truncation.
- Errors identify the record/column. Results contain generated row numbers,
  fixed class names and numeric **decision margins**, with a CSV download.
  User-provided text is never copied into result cells.
- Class 1: Billboard chart entry in HSP-S benchmark. Class 0: no chart entry in
  HSP-S benchmark. Neither is a future-success forecast.
- Positive/negative margin selects class 1/0. Margin magnitude is not calibrated
  probability or songwriting advice. Four included rows were seen in training.

## MP3 preview — working; classification unavailable

MP3 extension **and decoded format** are checked. Maximum 25 MiB, 480 seconds,
8–48 kHz, one/two channels, 192 MiB decoded float32 budget. The time/channel cap
is normally tighter than the byte budget. Decoding is incremental (16,384-frame
blocks), in a separate process, with a 25-second wall timeout and process-group
termination. Empty, silent, nonfinite, corrupt, spoofed and detectable truncated
files are rejected. Some playable truncations have no reliable length header;
this is not forensic recording-completeness verification.

Audio preview includes playback, duration, sample rate, channel count, RMS and
peak envelope. RMS uses channel energy so opposite-phase stereo is not mistaken
for silence. A permission checkbox precedes analysis. Uploaded bytes stay in
Streamlit session memory for playback; decoder files use isolated temporary
folders, are removed on success/failure, and are not logged or sent to APIs.
Clearing the upload/session releases application references to the bytes; this
is not a guarantee of forensic memory erasure or cleanup after machine power loss.

macOS/Linux are supported for supervised decoding; Windows fails with a clear
message. WAV/FLAC/model/pickle/ZIP uploads are not supported. The app only loads
its packaged, verified local model. No upload can replace a model or enable
MP3 classification. See [compatibility evidence](feature_compatibility.md).

## Verification

`tests/test_upload_csv.py`, `test_upload_audio.py`, `test_feature_compatibility.py`
and `test_upload_ui.py` cover these boundaries. The actual generated MP3 decoder
is exercised; mocked stream headers only exercise expensive boundary conditions.
Synthetic fixtures provide software evidence, never hit/non-hit accuracy.
