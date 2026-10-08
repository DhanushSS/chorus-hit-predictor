"""Explicit dispatch keeps historical and new-target score semantics separate."""
def load_for_task(task_id, run=None):
    if task_id == 'legacy_year_end':
        from .artifacts import load_run
        return load_run(run)
    if task_id == 'hit_nonhit_v1':
        from .hit_nonhit.artifacts import load
        if run is None:
            raise ValueError('A separately verified new-target run is required')
        return load(run, require_evaluated=True)
    raise ValueError('Unknown task; refusing to infer labels from a model filename')
