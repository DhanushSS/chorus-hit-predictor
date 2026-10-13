"""Predeclared nested group research. Never loads a locked feature/label table."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from chorus_hit.config import ROOT
from chorus_hit.hit_nonhit.common import read, write, sha, digest, environment
from chorus_hit.hit_nonhit.freeze_split import grouped_folds
from chorus_hit.hit_nonhit.train import fit
from chorus_hit.hit_nonhit.modeling import cluster_interval
from chorus_hit.hsp_s.data import development
from chorus_hit.hsp_s.modeling import metrics


def estimator(c, seed):
    scale = {'standard':StandardScaler(), 'robust':RobustScaler(),
             'quantile':QuantileTransformer(n_quantiles=200,output_distribution='normal',random_state=seed)}[c['scaler']]
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('variance',VarianceThreshold()),('scale',scale)]
    if c.get('select_percentile'):steps.append(('select',SelectPercentile(f_classif,percentile=c['select_percentile'])))
    svm=lambda: SVC(C=c.get('C',1),gamma=c.get('gamma','scale'),class_weight='balanced',max_iter=20000,cache_size=256)
    hist=lambda:HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=5,learning_rate=.08,early_stopping=False,random_state=seed)
    family=c['family']
    if family=='svm': model=svm()
    elif family=='hist': model=hist()
    elif family=='extra': model=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=3,class_weight='balanced',n_jobs=1,random_state=seed)
    else:model=VotingClassifier([('svm',svm()),('hist',hist()),('lr',LogisticRegression(C=.1,class_weight='balanced',max_iter=3000))],voting='hard',n_jobs=1)
    return Pipeline(steps+[('model',model)])


def run(split, out, config):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=read(config);frame,columns,_,sm=development(split)
    X=frame[columns];y=frame.label.to_numpy();g=frame.group.to_numpy();seed=cfg['seed']
    identity={'config_sha256':sha(config),'development_sha256':sha(Path(split)/'development.parquet'),
              'split_manifest_sha256':sha(Path(split)/'manifest.json'),'implementation_sha256':sha(__file__),
              'environment':environment(),'rows':len(y),'groups':len(set(g)),'features':len(columns),
              'class_counts':dict(Counter(str(v) for v in y)),'old_test_loaded':False}
    write(out/'identity.json',identity);trials=[]
    def fitting(c,tr,va,scope,labels=None):
        target=y if labels is None else labels
        if sum(t['seconds'] for t in trials)>cfg['max_trial_seconds']:raise RuntimeError('Declared research fit budget exhausted')
        start=time.monotonic();item={'candidate':c['id'],'scope':scope,'status':'failed'}
        try:
            model=estimator(c,seed);_,warnings=fit(model,X.iloc[tr],target[tr],cfg['per_fit_seconds'])
            p=model.predict(X.iloc[va]);item.update(status='complete',metrics=metrics(target[va],p),warnings=warnings)
        except (ValueError,RuntimeError,TimeoutError) as error:
            item['error']=str(error);model=None;p=None
        item['seconds']=time.monotonic()-start;trials.append(item);write(out/'trials'/f'{len(trials):04d}.json',item)
        print(scope,c['id'],item['status'],flush=True)
        return model,p,item
    def select(ix,scope):
        folds=grouped_folds(y[ix],g[ix],cfg['inner_folds'],seed);scores=[]
        for c in cfg['candidates']:
            result=[fitting(c,ix[tr],ix[va],f'{scope}/inner{i}')[2] for i,(tr,va) in enumerate(folds)]
            score=float(np.mean([r['metrics']['balanced_accuracy'] for r in result])) if all(r['status']=='complete' for r in result) else None
            scores.append({'candidate':c,'balanced_accuracy':score})
        valid=[r for r in scores if r['balanced_accuracy'] is not None]
        if not valid:raise RuntimeError('No candidate completed all inner fits')
        return sorted(valid,key=lambda r:(-r['balanced_accuracy'],r['candidate']['id']))[0]['candidate'],scores
    outer=grouped_folds(y,g,cfg['outer_folds'],seed);pred=np.empty(len(y),int);base=np.empty(len(y),int);folds=[]
    baseline=cfg['candidates'][0]
    for i,(tr,va) in enumerate(outer):
        chosen,comparison=select(tr,f'outer{i}')
        _,p,status=fitting(chosen,tr,va,f'outer{i}/selected')
        _,bp,bs=fitting(baseline,tr,va,f'outer{i}/baseline')
        if p is None or bp is None:raise RuntimeError('Outer fit failed; nested research incomplete')
        pred[va]=p;base[va]=bp;folds.append({'fold':i,'chosen':chosen,'metrics':status['metrics'],'baseline':bs['metrics'],'comparison':comparison})
    chosen,comparison=select(np.arange(len(y)),'final_development')
    # Diagnostics are predeclared; they cannot revise cohort or selection.
    codec={str(k):{'rows':len(v),'class_counts':dict(Counter(str(x) for x in v.label)),'groups':v.group.nunique()} for k,v in frame.groupby('nuisance_codec')}
    extractor={str(k):len(v) for k,v in frame.groupby('nuisance_extractor')}
    sizes=frame.groupby('group').size();pure=frame.groupby('group').label.nunique();largest=sizes.idxmax()
    diag={'codec':codec,'extractor':extractor,'group_size_quantiles':sizes.quantile([0,.5,.9,.99,1]).to_dict(),
          'pure_label_groups':int((pure==1).sum()),'mixed_label_groups':int((pure==2).sum()),
          'missing_cells':int(X.isna().sum().sum()),'duplicate_feature_rows':int(X.duplicated().sum()),
          'constant_features':int((X.nunique(dropna=True)<=1).sum()),'release_era':'Unavailable in frozen development metadata; no positional join attempted',
          'genre':'Unavailable in frozen development metadata; not inferred from labels or fetched through test records'}
    def control(ix,scope):
        p=np.empty(len(ix),int)
        for j,(tr,va) in enumerate(grouped_folds(y[ix],g[ix],3,seed)):
            _,pv,r=fitting(baseline,ix[tr],ix[va],f'{scope}/{j}')
            if pv is None:return {'status':'failed'}
            p[va]=pv
        return {'rows':len(ix),'groups':len(set(g[ix])),'metrics':metrics(y[ix],p),'interval':cluster_interval(y[ix],p,g[ix],seed)}
    diag['mp3_only']=control(np.flatnonzero(frame.nuisance_codec.to_numpy()=='mp3'),'mp3_control')
    diag['without_largest_group']=control(np.flatnonzero(g!=largest),'largest_group_control')
    rng=np.random.default_rng(seed);nulls=[];importance=[]
    # Global permutations test the no-association null, NOT an exchangeable within-artist null.
    for repeat in range(3):
        shuffled=rng.permutation(y);p=np.empty(len(y),int)
        for j,(tr,va) in enumerate(grouped_folds(y,g,3,seed)):
            _,pv,record=fitting(baseline,tr,va,f'global_null/{repeat}/{j}',labels=shuffled)
            if pv is None:raise RuntimeError('Global-null fit failed')
            p[va]=pv
        nulls.append(metrics(shuffled,p)['balanced_accuracy'])
    diag['global_permutation_balanced_accuracy']=nulls
    for i,(tr,va) in enumerate(grouped_folds(y,g,3,seed)):
        model,p,status=fitting(baseline,tr,va,f'family_importance/{i}')
        if model is None:continue
        reference=status['metrics']['balanced_accuracy'];drops={}
        for family in ['lowlevel.','rhythm.','tonal.']:
            altered=X.iloc[va].copy();cols=[c for c in columns if c.startswith(family)];order=rng.permutation(len(va))
            altered.loc[:,cols]=altered[cols].to_numpy()[order]
            drops[family]=reference-metrics(y[va],model.predict(altered))['balanced_accuracy']
        importance.append(drops)
    diag['feature_family_permutation_drop']=importance
    result={'status':'DEVELOPMENT_ONLY','identity':identity,'selected_candidate':chosen,'nested':metrics(y,pred),
            'nested_interval':cluster_interval(y,pred,g,seed),'fixed_baseline':metrics(y,base),
            'fixed_baseline_interval':cluster_interval(y,base,g,seed),'outer_folds':folds,'final_comparison':comparison,
            'diagnostics':diag,'trials':trials,'external_test':'NOT_AVAILABLE','promotion':'NONE',
            'target_status':'80% observed on development only' if metrics(y,pred)['accuracy']>=.8 else '80% not reached'}
    write(out/'results.json',result);write(out/'development_predictions.json',{'uuid':frame.uuid.tolist(),'groups':g.tolist(),'labels':y.tolist(),'nested':pred.tolist(),'baseline':base.tolist()})
    print(json.dumps({k:result[k] for k in ['status','selected_candidate','nested','target_status']},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--split',required=True);p.add_argument('--out',required=True);p.add_argument('--config',default='configs/research_v2.json');a=p.parse_args();run(a.split,a.out,a.config)
