"""Declared grouped folds, honest metrics, uncertainty, and strict target checks."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

TARGET_METRICS=('accuracy','balanced_accuracy','precision','recall','f1')


def binary_vector(value, name):
    """Accept numeric exact 0/1 (including floats); reject bools and strings."""
    a = np.asarray(value)
    if a.ndim != 1 or not a.size or a.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a nonempty 1D numeric binary vector (no bools/strings)')
    if not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
        raise ValueError(f'{name} must contain finite exact binary values 0 or 1')
    return a.astype(int)


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def bootstrap_inputs(y, predictions, groups, repeats, scores=None):
    positive_integer(repeats, 'repeats')
    y = binary_vector(y, 'labels')
    ps = [binary_vector(p, 'predictions') for p in predictions]
    if any(p.shape != y.shape for p in ps):
        raise ValueError('Bootstrap labels/predictions must be aligned')
    g = np.asarray(groups)
    if g.ndim != 1 or g.shape != y.shape or pd.isna(g).any():
        raise ValueError('Bootstrap groups must be aligned 1D non-null IDs')
    if any(not isinstance(v, (str, int, float, np.integer, np.floating)) or
           isinstance(v, (bool, np.bool_)) or (isinstance(v, str) and not v.strip()) or
           (not isinstance(v, str) and not np.isfinite(v)) for v in g):
        raise ValueError('Invalid bootstrap group ID')
    # Factorization also permits mixed string/numeric IDs without sorting them.
    g, _ = pd.factorize(g, sort=True)
    if len(np.unique(g)) < 2 or len(np.unique(y)) < 2:
        raise ValueError('Bootstrap requires two classes and at least two groups')
    s = None if scores is None else np.asarray(scores, dtype=float)
    if s is not None and (s.shape != y.shape or not np.isfinite(s).all()):
        raise ValueError('Invalid score shape or values')
    return y, ps, g, s


def classification_metrics(y, prediction, scores=None):
    y=binary_vector(y, 'labels'); p=binary_vector(prediction, 'predictions')
    if len(y)==0 or len(y)!=len(p) or not set(y).issubset({0,1}) or not set(p).issubset({0,1}):
        raise ValueError('Binary nonempty aligned labels/predictions are required')
    tn=int(np.sum((y==0)&(p==0))); fp=int(np.sum((y==0)&(p==1)))
    fn=int(np.sum((y==1)&(p==0))); tp=int(np.sum((y==1)&(p==1)))
    r0=tn/(tn+fp) if tn+fp else None; r1=tp/(tp+fn) if tp+fn else None
    precision=tp/(tp+fp) if tp+fp else 0.; recall=r1 if r1 is not None else 0.
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    f0=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    auc=None
    if scores is not None:
        scores=np.asarray(scores,float)
        if scores.shape!=y.shape or not np.isfinite(scores).all(): raise ValueError('Invalid score shape or values')
        if r0 is not None and r1 is not None: auc=float(roc_auc_score(y,scores))
    return {'accuracy':(tn+tp)/len(y),'balanced_accuracy':(r0+r1)/2 if r0 is not None and r1 is not None else None,
        'precision':precision,'recall':recall,'f1':f1,'class_0_recall':r0,'macro_f1':(f0+f1)/2,
        'roc_auc':auc,'confusion_matrix':[[tn,fp],[fn,tp]],'n':len(y),'support':{'0':tn+fp,'1':tp+fn},
        'undefined':(["balanced_accuracy","roc_auc"] if len(set(y))<2 else []) +
                    (["precision_zero_division_defined_as_zero"] if tp+fp==0 else [])}


def score_model(model, X):
    if list(model.classes_) != [0,1]: raise ValueError('Unsupported class mapping; expected ordered [0,1]')
    if hasattr(model,'decision_function'):
        scores=np.asarray(model.decision_function(X)); kind='decision_margin'; threshold=0.
    elif hasattr(model,'predict_proba'):
        scores=np.asarray(model.predict_proba(X))[:,1]; kind='uncalibrated_class_1_probability'; threshold=.5
    else: raise ValueError('Estimator does not expose a supported continuous score')
    if scores.ndim!=1 or not np.isfinite(scores).all(): raise ValueError('Unsupported/nonfinite scores')
    return scores,{'kind':kind,'threshold':threshold,'positive_class':1,'calibrated':False,
                   'prediction_rule':'estimator.predict (native tie handling)'}


def target_status(metrics,evaluation_status):
    checks={k:bool(metrics.get(k) is not None and metrics[k]>.75) for k in TARGET_METRICS}
    fresh=evaluation_status=='fresh_test'
    return {'threshold':.75,'comparison':'strictly_greater','evaluation_status':evaluation_status,
        'metrics':{k:{'value':metrics.get(k),'passed':v} for k,v in checks.items()},
        'target_met_in_this_evaluation':all(checks.values()),'fresh_test_available':fresh,
        'fresh_test_target_met':all(checks.values()) if fresh else None}


def grouped_folds(frame,n_splits,seed,groups,features):
    y=frame.label.to_numpy(); groups=np.asarray(groups)
    if len(np.unique(groups))<n_splits or n_splits<2: raise ValueError('Invalid group support for requested folds')
    for label in (0,1):
        if len(np.unique(groups[y==label]))<n_splits: raise ValueError('Each class needs support across at least n_splits groups')
    result=list(StratifiedGroupKFold(n_splits,shuffle=True,random_state=seed).split(frame[features],y,groups))
    for fit,val in result:
        if set(groups[fit])&set(groups[val]): raise ValueError('Group overlap')
        if len(set(y[fit]))!=2 or len(set(y[val]))!=2: raise ValueError('One-class fold; change the declared protocol, not its seed opportunistically')
        if set(frame.track_id.iloc[fit])&set(frame.track_id.iloc[val]): raise ValueError('Recording identity overlap')
        if set(pd.util.hash_pandas_object(frame[features].iloc[fit],index=False)) & set(pd.util.hash_pandas_object(frame[features].iloc[val],index=False)):
            raise ValueError('Exact duplicate features cross partitions')
    return result


def bootstrap_intervals(y,prediction,groups,scores=None,repeats=2000,seed=42):
    y, (p,), g, s = bootstrap_inputs(y, [prediction], groups, repeats, scores)
    unique=np.unique(g); rng=np.random.default_rng(seed)
    keys=[*TARGET_METRICS,'class_0_recall','macro_f1','roc_auc']; samples={k:[] for k in keys}; skipped=0
    for _ in range(repeats):
        ix=np.concatenate([np.flatnonzero(g==v) for v in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ix]))!=2: skipped+=1; continue
        m=classification_metrics(y[ix],p[ix],None if s is None else s[ix])
        for k in keys:
            if m[k] is not None: samples[k].append(m[k])
    return {'intervals':{k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in samples.items()},
        'method':'percentile bootstrap of whole artist-string groups; fixed predictions, conditional on fitted procedure',
        'requested_resamples':repeats,'valid_resamples':repeats-skipped,
        'unavailable_reason':'No two-class resamples' if skipped==repeats else None,'one_class_resamples_skipped':skipped,'groups':len(unique),
        'limitations':'Few groups and model-selection uncertainty are not fully captured; nested folds are dependent.'}


def paired_bootstrap(y,p1,p0,groups,repeats=2000,seed=42):
    y, (p1, p0), g, _ = bootstrap_inputs(y, [p1,p0], groups, repeats)
    unique=np.unique(g)
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(repeats):
        ix=np.concatenate([np.flatnonzero(g==v) for v in rng.choice(unique,len(unique),True)])
        if len(set(y[ix]))<2: continue
        draws.append(classification_metrics(y[ix],p1[ix])['balanced_accuracy']-classification_metrics(y[ix],p0[ix])['balanced_accuracy'])
    return {'balanced_accuracy_difference':classification_metrics(y,p1)['balanced_accuracy']-classification_metrics(y,p0)['balanced_accuracy'],
            'ci95':np.quantile(draws,[.025,.975]).tolist() if draws else None,
            'unavailable_reason':None if draws else 'No two-class resamples','valid_resamples':len(draws),'paired_by':'same rows and resampled whole groups'}
