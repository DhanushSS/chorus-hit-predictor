# Chorus extraction pilot — 278-song cohort

**NOT RUN — NO AUDIO** (10 October 2026).

No 10–20-song pilot could begin: there are zero permitted exact recordings for
either class. Real excerpts extracted: 0. Human-reviewed choruses: 0 hits / 0
non-charting candidates. Decoder/extraction failures on real files: 0 attempts.
No recording match, permission, timestamp or human review was invented.

The existing repetition selector remains the baseline. Pychorus was not installed
or compared because there were no real recordings on which to assess it.
The unchanged contract is mono 22,050 Hz, exactly 15 seconds (330,750 samples),
518 ordered librosa features for both classes, `shared-librosa-518-v2`.
Automatic fallback excerpts remain explicitly unverified.

## What was tested

The full automated suite includes generated-waveform checks for stereo resampling,
feature dimensions/finiteness, excerpt length, annotations, corrupt codecs, silence,
NaN/infinity, duration/bounds and path traversal. New intake regressions reject
negative/nonfinite chorus positions and records outside the fixed 278 list.
These tests validate software behavior; generated tones are not songs or evidence
of successful chorus detection on this cohort.

## Real pilot procedure once permitted files exist

1. Use the generated local intake; keep IDs and proposed labels unchanged.
2. Select 5–10 tracks per class, prioritizing complete same-album matched support
   and several independent artists. Record the actual source and rights basis.
3. Verify studio recording/version, codec, duration and duplicates. Run intake
   `--dry-run` first. Missing files and controls lacking positive audio support are
   reported rather than silently replaced.
4. Run the unchanged detector/extractor, inspect silence/clipping/source artifacts,
   and listen to proposed 15-second excerpts. Record actual start times and the
   name of the person who listened. Leave `chorus_confirmed` blank until checked.
5. At least five genuine human-reviewed choruses per class are required to freeze
   a real split. Larger artist-group support must also pass the existing guards.

The acquisition blocker and permitted feature alternatives are documented in
[audio feasibility](audio_feasibility_278.md). No new accuracy is available.
