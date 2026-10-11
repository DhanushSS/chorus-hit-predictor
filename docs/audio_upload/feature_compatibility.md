# MP3 feature compatibility

**Decision: preview only; no audio classification.** The current SVM expects 436
whole-recording scalar HSP-S descriptors. The archived 518 librosa chorus features
are a different representation and cannot be substituted.

## Actual experiment

On 11 October 2026, an isolated Python 3.12 environment on Apple Silicon macOS
successfully installed `essentia==2.1b6.dev1389` and extracted a software-generated
15-second two-tone MP3. A bounded repeat is recorded in [macos_probe.json](macos_probe.json).
This is a real decode/extraction software check, **not a song, a matched reference,
or an accuracy measurement**. No copyrighted recordings were obtained.

| Check | Evidence |
| --- | --- |
| Source extractor | All 6,235 frozen development rows report Essentia `2.1-beta2` |
| Tested package | `2.1b6.dev1389`; native version `2.1-beta6-dev`; extractor `music 2.0` |
| Exact scalar coverage | **381 / 436** on the generated fixture |
| Missing | 54 rhythm histogram statistics plus `tonal.key_strength` |
| Identical-recording reference pairs | **0** |
| Numeric / distribution parity | **NOT DEMONSTRATED** |
| Independent labelled MP3 evaluation | **NOT AVAILABLE** |
| App audio prediction | **Disabled**, including if a single numeric pair passes |

The 54 missing fields are nine statistics for each of the first/second BPM
histogram peak's BPM, spread and weight. The modern extractor exposes different
scalar rhythm outputs and multiple key profiles. Repeating a scalar into nine
statistics, selecting a key profile by guess, renaming fields or zero-filling
would fabricate compatibility. Full missing-key list and parameters are in the
probe receipt. Coverage may also vary for short, silent or unusual recordings.

## Reproducible probe profile

`chorus_hit/audio_upload/extract_essentia.py` pins the package and explicitly sets:
44,100 Hz analysis; low-level 2,048-frame/1,024-hop; tonal 4,096-frame/2,048-hop;
Blackman-Harris 62 windows; zero padding 0; silent-frame noise mode;
Degara rhythm method, 40–208 BPM; full-file time range; loudness frame 88,200,
hop 44,100. Low-level/rhythm/tonal aggregation is mean, variance, median, minimum,
maximum, first/second derivative mean and variance. MFCC/GFCC vector outputs are
excluded from this scalar contract. Native decoder/downmix and modern key-profile
implementations remain package defaults; historical equivalence is **unknown**.
BPM is beats/minute; frame and hop lengths are samples; descriptor units otherwise
follow the named native algorithm. No unit conversion is guessed.

The exact beta2 publisher profile, downmix conventions, original masters and
algorithm-level distribution agreement have not been recovered. Therefore the
new profile is explicitly labelled **research-unvalidated**, not a reconstruction
of the historical extractor.

Install separately; do not change the frozen model environment:

```bash
python3.12 -m venv .venv-audio
.venv-audio/bin/python -m pip install -r requirements-audio-probe.txt
.venv-audio/bin/python scripts/probe_essentia.py --out output/audio_probe.json
```

Create `output/` first if absent. The generated MP3 is temporary and removed on
success or failure. The probe has a 90-second process-group timeout. This CLI is
for the generated fixture; the Streamlit app never runs Essentia on uploads.
The core app uses SoundFile for its bounded MP3 preview and works without Essentia.
macOS installation/execution is locally verified; Linux has a separate CI probe
and receipt. Native Windows extraction and supervised MP3 preview are unsupported.

## Future parity gate

For lawfully analyzable identical masters, compare exact ordered keys using
`scripts/check_audio_feature_parity.py`. Its predeclared numerical screening
thresholds are `rtol=1e-4, atol=1e-6` per field; it requires finite scalars and
rejects missing fields. These are provisional engineering tolerances, not an
established scientific equivalence criterion. Review recording identity, codec,
extractor profile and family-wise drift on multiple representative reference
pairs before deciding a defensible tolerance/cohort protocol. A single passing
pair never enables inference. Independent labelled audio evaluation and source
shift checks are also missing. The historical 73.17% cannot be assigned to MP3s.

**Required input to continue parity work:** permitted same-master reference
recordings with matching original 436-feature vectors, recording identity and
extraction-profile provenance. No other completed work depends on this input.

## Primary sources

- [Essentia MusicExtractor documentation](https://essentia.upf.edu/tutorial_extractors_musicextractor.html)
  describes frame-level and aggregate descriptor outputs.
- [Streaming music extractor](https://essentia.upf.edu/streaming_extractor_music.html)
  documents the extractor architecture/profile.
- [AcousticBrainz project closure](https://musicbrainz.wordpress.com/2022/02/16/acousticbrainz-making-a-hard-decision-to-end-the-project/)
  explains the end of new submissions; no new-submission API is assumed.
