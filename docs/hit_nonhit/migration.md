# Separate-task migration

The 2026-10-08 user request authorizes a new research target on a separate branch. Earlier faculty
approval concerns the old adaptation. New approval is pending; no historical labels are changed.

| Aspect | Preserved project | New research namespace |
|---|---|---|
| Target | Year-end hit vs other weekly-chart song | Weekly Hot 100 vs verified non-charted through cutoff |
| Data | `data/chorus_features.csv` | `output/hit_nonhit/v1/<dataset>/` |
| Extractor | `legacy-librosa-518-v1` | `shared-librosa-518-v2`, both new classes |
| Active run | `configs/active_run.json` unchanged | No active/default change |
| Results | `results/v2/v4_std74_001/` unchanged | Ignored `results/hit_nonhit/v1/<run>/` for genuine authorized outputs |
| Split | Previously exposed 154-song historical split | New locked connected-artist split |
| Inference | `python -m chorus_hit.predict` | `python -m chorus_hit.hit_nonhit.predict --run ... --audio ...` |

`chorus_hit.task_artifacts.load_for_task` provides explicit programmatic dispatch. New-target loader
rejects legacy artifacts and synthetic fixtures; old validators are preserved. Streamlit shows a useful
blocked-data section until a compatible genuine evaluated bundle is present, then an optional research
view. The existing app model and all saved data, models, results, reports and slides are preserved.

No old waveform-parity claim is borrowed for the new extractor. No new metrics are substituted for
historical metrics. Local raw data, copyright-limited evidence and synthetic model outputs stay out of Git.
