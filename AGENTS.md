# Chorus project guidance

Read `docs/AUDIT_BRIEF.md` and `docs/experiment_log.md` before new experiments. The original instructions are preserved; this file is a short entry point.

- Preserve V1 data, fitted model, split membership and published evidence. New work belongs on a reviewable branch. Do not push/merge/publish without user authorization.
- Use `.venv/bin/python` and exact requirements. Run `python -m pytest -q` for integrity checks. Fast checks: `python -m pytest -q -m 'not integration'`.
- Use new run IDs and immutable run folders. Resume only an incomplete run with matching code/config/data identities.
- Keep every learned transform inside grouped training folds. Original test membership is historical; a reshuffle is not fresh data.
- All scores, failed trials, warnings and blocked inputs belong in the run manifest and experiment log. Never impose a fake empirical test threshold.
- Use the common validated artifact loader for app, CLI and reports. Preserve extraction versions and use matched audio for future feature changes.
- End with the next unblocked action. See `docs/REPRODUCE_V2.md` for supported commands.
