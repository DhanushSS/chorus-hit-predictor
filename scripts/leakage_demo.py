"""Exploratory all-data comparison of random-song and normalized artist-name splits.

Includes the 154 historical rows. This is not fresh confirmation or a performance
ceiling. Existing results/leakage_demo.csv is preserved as historical evidence.
"""
import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from chorus_hit.audit import load_dataset, LEGACY_SHA256, atomic_json
from chorus_hit.config import FEATURE_COLUMNS
from chorus_hit.data import normalize_artist
from chorus_hit.evaluation import grouped_folds, positive_integer
from chorus_hit.contracts import frozen_split
from chorus_hit.diagnostics import identity, bounded_command


def models(workers=1):
    return {
        'Logistic regression':make_pipeline(StandardScaler(),PCA(.95),LogisticRegression(C=.1,max_iter=3000)),
        'RBF SVM':make_pipeline(StandardScaler(),SVC(C=1)),
        'Random forest':RandomForestClassifier(500,random_state=0,n_jobs=workers),
        'Extra trees':ExtraTreesClassifier(500,random_state=0,n_jobs=workers),
        'Hist gradient boosting':HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,random_state=0),
    }


def validate_settings(repeats, seed, workers):
    positive_integer(repeats,'repeats');positive_integer(workers,'workers')
    if repeats>10 or workers>4: raise ValueError('Diagnostic maximum: 10 repeats, 4 workers')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32-repeats: raise ValueError('Invalid seed')


def compare(frame,features,seeds,estimators,output):
    """Persist each completed model/split/repeat before continuing (no rounding)."""
    records=[]; groups=frame.artist.map(normalize_artist).to_numpy()
    for name,model in estimators.items():
        for seed in seeds:
            splits={'random_song_with_artist_overlap':list(StratifiedKFold(5,shuffle=True,random_state=seed).split(frame[features],frame.label)),
                    'normalized_artist_name_grouped':grouped_folds(frame,5,seed,groups,features)}
            for protocol,folds in splits.items():
                row={'model':name,'seed':seed,'protocol':protocol,
                     'folds':[{'fit':frame.track_id.iloc[f].tolist(),'validation':frame.track_id.iloc[v].tolist()} for f,v in folds]}
                caught=[]
                try:
                    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        pred=cross_val_predict(model,frame[features],frame.label,cv=folds,n_jobs=1)
                        row.update(status='complete',balanced_accuracy=float(balanced_accuracy_score(frame.label,pred)))
                except Exception as exc:
                    row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                row['warnings']=[f'{type(w.message).__name__}: {w.message}' for w in caught]
                records.append(row);atomic_json(Path(output)/'records.json',records)
                if row['status']=='failed': raise RuntimeError(row['error'])
    pd.DataFrame([{k:v for k,v in r.items() if k not in {'folds','warnings'}} for r in records]).to_csv(Path(output)/'scores.csv',index=False)
    return records


def diagnostic_worker(output,repeats,seed,workers):
    validate_settings(repeats,seed,workers)
    ctx=load_dataset(expected_sha256=LEGACY_SHA256); _,dev,historical=frozen_split(ctx.frame)
    atomic_json(Path(output)/'protocol.json',identity(__file__,status='exploratory_all_data',
        fresh_test=False,historical_rows_included=historical,all_track_ids=ctx.frame.track_id.tolist(),
        repeats=repeats,seeds=list(range(seed,seed+repeats)),workers=workers,folds=5,
        note='Inspected songs; changing seeds does not create independent evidence'))
    return compare(ctx.frame,FEATURE_COLUMNS,range(seed,seed+repeats),models(workers),output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--timeout-seconds',type=float,default=300)
    p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();validate_settings(a.repeats,a.seed,a.workers)
    if a._worker: diagnostic_worker(a.out,a.repeats,a.seed,a.workers)
    else:
        command=[sys.executable,'-m','scripts.leakage_demo','--out',str(a.out.resolve()),
                 '--repeats',str(a.repeats),'--seed',str(a.seed),'--workers',str(a.workers),'--_worker']
        bounded_command(command,a.out,a.timeout_seconds)


if __name__=='__main__':main()
