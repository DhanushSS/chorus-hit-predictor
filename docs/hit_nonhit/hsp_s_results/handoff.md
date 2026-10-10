# HSP-S completion and verification — 10 October 2026

## Summary

The approved fallback is implemented, genuinely trained and evaluated. Its single
locked test produced **73.07% accuracy and 73.17% balanced accuracy**. The 95%
connected-group bootstrap interval for balanced accuracy is **70.20–76.08%**
(1,000 resamples, conditional on this dataset and fixed trained model).

This result meets the requested 70–75% range on the HSP-S benchmark. It does not
establish 75% performance, future-hit prediction, or the accuracy of a 15-second
chorus model. The original 278-song task remains blocked by permitted audio.

## Deliverables and status

| Deliverable | Verified outcome |
| --- | --- |
| Published feature source | 7,736 HSP-S records; full publisher MD5 verified |
| Accepted dataset | 7,728 songs: 3,865 chart entries / 3,863 without chart entries |
| Exclusions | 4 conflicting feature rows, 2 identical feature duplicates, 2 conflicting recording labels |
| Feature schema | 436 scalar Essentia audio descriptors per accepted song |
| Independence groups | 2,633 connected groups; artist, recording and album links plus duplicate checks |
| Development partition | 6,235 songs, 2,106 groups; five outer / three inner folds |
| Locked test | 1,493 songs, 527 groups; used once after selection |
| Model comparison | 27 configurations; 486 inner validation fits completed, zero failed |
| Selected model | RBF SVM, C=1.0, balanced class weights, gamma=scale, all scalar features |
| Preprocessing | Median imputation, constant removal, standardization; fitted within each training fold |
| Training balanced accuracy | 79.55%; optimistic in-sample assessment |
| Nested development balanced accuracy | 70.94%; estimates the model-selection procedure |
| Locked accuracy / balanced accuracy | 73.07% / 73.17% |
| Locked hit precision / recall / F1 | 70.42% / 75.35% / 72.80% |
| Locked ROC-AUC / average precision | 0.8076 / 0.7900 |
| Test confusion matrix | TN=553, FP=226, FN=176, TP=538 |
| Local prediction demo | Real four-row feature CSV uploaded and classified successfully through Streamlit |
| Existing project | All 54 protected artifact hashes unchanged; active chorus model preserved |
| Exact-278 recordings / chorus records | 0 / 0; no new 518-feature chorus model trained |

The model's decision threshold is the native SVM margin of zero. Its scores are
uncalibrated margins, not percentages or probabilities. The selected final fit
took 2.46 seconds; inner validation fits accumulated 1,226.05 seconds. These are
local timings, excluding acquisition and the rest of the workflow, and are not
comparative speed claims. Thirty-eight trial fits emitted convergence warnings;
all were retained. The selected final fit emitted no warnings.

## Bias diagnostics and limits

No predefined blocking flag fired. However, this is **not proof of no dataset
bias**. A development-only classifier using codec, extractor version, sample rate
and duration reached **61.08% balanced accuracy**, below the preregistered 65%
blocking threshold but above chance.

Within-group label permutations scored **65.98%, 66.48% and 66.00%**. These
permutations preserve group-level label composition, and only 412 mixed-label
groups (3,177 development songs) permit label exchanges. Therefore this is not a
global random-label null test; strong residual artist/cohort structure remains.
Its mean stayed below the declared blocking boundary (nested balanced accuracy
minus two percentage points, with a 60% floor). We did not change that rule after
seeing results. These diagnostics limit any claim that acoustic content alone
causes or predicts chart success.

The source's positive/negative labels reflect publisher chart matching through
6 July 2019. Missing chart entries are not independently verified lifetime
non-charting status. The largest identity group contains 636 songs. Grouping can
only use known identities and does not resolve every alias or cover. Results
cannot be directly ranked against Eric Liu or the original chorus dataset.

## Tests and application verification

Model implementation was committed **before training** at
`49100549c94bcedef3061390ab607cc15a781d9a`.

| Check executed | Outcome |
| --- | --- |
| `.venv/bin/python -m pytest tests/hit_nonhit/test_hsp_s.py -q` | 17 passed, 1.15 seconds |
| `.venv/bin/python -m pytest -q` | 285 passed, 4 existing warnings, 154.99 seconds locally |
| `.venv/bin/python -m pip check` | No broken requirements |
| `.venv/bin/python -m compileall -q chorus_hit/hsp_s` | Passed |
| `git diff --check` | Passed |
| Full source preparation and manifest verification | Passed; 7,728 accepted records |
| Real nested training | All 27 candidates completed all required inner fits |
| One-use locked evaluation | Completed, access ledger retained |
| Real model load and CSV prediction API | Four development records classified; 0.113 seconds including manifest/model loading |
| Streamlit `AppTest.from_file(...).run()` on the evaluated feature page | No exceptions; displayed 73.1% accuracy, 73.2% balanced accuracy, 1,493 test songs |
| Browser upload and classify action | Four prediction rows visibly returned, matching the API outputs |
| SHA-256 preservation check | 54/54 protected files matched |
| GitHub Project integrity workflow on the implementation commit | SUCCESS; 285 passed, 4 warnings, 406.41 seconds |

Exact remote evidence: [workflow run 38046487532](https://github.com/DhanushSS/chorus-hit-predictor/actions/runs/38046487532).
CI runs the regression suite and a separate synthetic pipeline smoke test; it
does not download HSP-S or reproduce the reported research training. Research
results above are from the verified local full-source run. The reporting script
was executed after evaluation and its output figures were visually inspected.

Screenshots of the successful upload and prediction are retained locally in
`output/hsp_s/`. They are not part of the research test-set evaluation. The demo
used four development records to verify the application, not to estimate accuracy.

## Files, reproduction and GitHub

- Pipeline: `chorus_hit/hsp_s/`; configuration: `configs/hsp_s_acousticbrainz.json`.
- Separate Streamlit page: `pages/1_Hit_vs_NonHit_Features.py`.
- Regression tests: `tests/hit_nonhit/test_hsp_s.py`.
- Report exporter: `scripts/report_hsp_s.py`.
- [Full metrics and figures](README.md), [machine-readable aggregates](results.json),
  [protocol and reproduction commands](../hsp_s_protocol.md).

To produce a report in a fresh destination after training/evaluation:

```bash
python scripts/report_hsp_s.py \
  --split output/hsp_s/split_001 --run results/hsp_s/run_001 \
  --out output/hsp_s/report_copy \
  --ci-url https://github.com/DhanushSS/chorus-hit-predictor/actions/runs/38046487532
python -m streamlit run app.py
```

The report exporter reads saved evaluation results; it does not open the locked
test again. In Streamlit, select **Hit vs Non-Hit feature study** for this model.
The original audio-upload model remains on the main page.

Code and safe aggregate reports belong to `feat/hit-vs-nonhit-v1`. The repository
is public. PR #6 remains closed and unmerged. No merge or reopening was performed.
The trained HSP-S bundle and detailed trial artifacts are local in ignored
`results/hsp_s/run_001/`; source/split files are in ignored `output/hsp_s/`.
**A fresh GitHub clone requires following the reproduction commands to obtain
this model.** No copyrighted recordings, credentials, raw source tables or private
absolute paths are included in the published report.

## Remaining blockers

1. **Original 278-song chorus objective:** still no permitted exact recordings.
   Completion needs downloadable recordings with a source/permission basis for
   local analysis, or compatible licensed 15-second chorus features with matching
   recording identities and the extraction specification. A Hugging Face account
   or Spotify plugin alone does not supply these permissions or feature matches.
2. **Verified non-hit claim:** the approved HSP-S fallback uses publisher labels.
   Independent absence verification would require complete, scoped chart coverage
   and reliable recording/artist matching; it is not implied by missing fields.
3. **New audio predictions with HSP-S:** the existing librosa chorus extractor is
   incompatible with these whole-recording Essentia features. The new demo accepts
   matching feature CSVs only. It must not apply this 73.17% score to MP3 uploads.

No user input is needed to use or reproduce the completed HSP-S benchmark.
