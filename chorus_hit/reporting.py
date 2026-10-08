"""Run-derived wording shared by reports and notebooks, with no frozen conclusions."""
from .artifacts import load_run


def summary_text(summary):
    name=summary['model_name'];n=summary['feature_count'];metric=summary['metrics']['balanced_accuracy'];ci=summary['ci95'].get('balanced_accuracy')
    status=summary['evaluation_status'].replace('_',' ')
    if ci is None: uncertainty='No uncertainty interval is available.'
    elif ci[0]>.5:uncertainty=f'The approximate 95% group-bootstrap interval ({ci[0]:.1%}-{ci[1]:.1%}) lies above 50% in this evaluation.'
    elif ci[1]<.5:uncertainty=f'The approximate 95% group-bootstrap interval ({ci[0]:.1%}-{ci[1]:.1%}) lies below 50% in this evaluation.'
    else:uncertainty=f'The approximate 95% group-bootstrap interval ({ci[0]:.1%}-{ci[1]:.1%}) includes 50%; a reliable advantage over chance is not established.'
    passed=summary['target_status']['target_met_in_this_evaluation']
    conclusion=f'The five-metric target is {"met" if passed else "not met"} in this evaluation. Fresh-test confirmation is {"available" if summary["target_status"]["fresh_test_available"] else "unavailable"}.'
    return {'title':f'{name}: {status}','method':f'The model accepts {n} ordered raw audio features. Selection criterion: {summary["selection_metric"]}.',
        'result':f'{name} records {metric:.1%} balanced accuracy for {status}.','uncertainty':uncertainty,'conclusion':conclusion}


def report_context(run_id):
    a=load_run(run_id);s=a.summary
    return {'summary':s,'text':summary_text(s),'manifest':a.manifest,'historical':None,'data':a.data}
