# Research execution log

Run 001 was interrupted before completion to fix incomplete diagnostic fit
logging found during code review. 10 completed trial records are retained locally and published in
`aborted_run_001.json`; the in-progress fit at interruption has no claimed score. No
outer/test result informed a configuration change. Run 002 uses the identical
registered grid, seed and development split, and now records every null-control
fit, warning and wall time under the same trial budget. No baseline evidence or
holdout access was changed.

Run 002 completed 226 recorded fits, with no failures or warnings. Its final
selection was hard voting. No new holdout was acquired and no model promoted.
