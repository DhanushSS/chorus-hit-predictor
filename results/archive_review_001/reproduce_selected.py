"""Independent, fixed-candidate reproduction of the supplied ZIP's claims.
No supplied Python or joblib is executed. No search or model promotion is performed.
"""
from pathlib import Path
import sys,json,hashlib,re,unicodedata,time,warnings,platform
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from threadpoolctl import threadpool_limits
from chorus_hit.config import FEATURE_COLUMNS
from chorus_hit.data import normalize_artist
from chorus_hit.train import metrics,artist_bootstrap
from chorus_hit.artifacts import load_run
from chorus_hit.evaluate_run import load_historical
OUT=Path(__file__).resolve().parent
if (OUT/'manifest.json').exists():raise FileExistsError('Completed review is immutable')
ARCHIVE=ROOT.parents[1]/'work/incoming_accuracy_review_20261003'
INCOMING=ARCHIVE/'project/chorus-hit-predictor'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
inputs={name:{'incoming_sha256':sha(INCOMING/name),'current_sha256':sha(ROOT/name),'identical':sha(INCOMING/name)==sha(ROOT/name)} for name in ['data/chorus_features.csv','results/split_manifest.csv','models/selected_model.joblib','requirements.txt']}
assert all(x['identical'] for x in inputs.values())
d=load_run('v1_baseline').data.copy()
split=pd.read_csv(ROOT/'results/split_manifest.csv').set_index('track_id')
g=d.artist.map(normalize_artist)
tkey=lambda t:re.sub(r'[^a-z0-9]+','',re.sub(r'[\(\[].*?[\)\]]',' ',unicodedata.normalize('NFKC',str(t)).casefold()))
assert not pd.DataFrame({'artist':g,'title':d.title.map(tkey)}).duplicated().any()
assert not d[FEATURE_COLUMNS].duplicated().any()
test=d.track_id.map(split.partition).eq('test').to_numpy()
cols=[c for c in FEATURE_COLUMNS if c.rsplit('_',2)[1]=='std'];assert len(cols)==74
tr=d.loc[~test].reset_index(drop=True);te=d.loc[test].reset_index(drop=True)
X=tr[cols].to_numpy(float);y=tr.label.to_numpy();groups=tr.artist.map(normalize_artist).to_numpy()
assert not set(groups)&set(te.artist.map(normalize_artist))
configuration={'purpose':'Reproduce only the supplied fixed winner; not a new search or unbiased evaluation','target':'year_end_vs_other','features':cols,'C':.1,'class_weight':'balanced','max_iter':3000,'quantiles':100,'quantile_seed':0,'cv_seeds':[100,101,102],'folds':5,'fits':16,'historical_partition':'Original 154 rows, already examined previously','no_promotion':True,'inputs':inputs}
dump(OUT/'config.json',configuration)
pipeline=make_pipeline(SimpleImputer(strategy='median'),VarianceThreshold(),QuantileTransformer(n_quantiles=100,output_distribution='normal',random_state=0),LogisticRegression(C=.1,class_weight='balanced',max_iter=3000))
started=time.time();folds=[];predrows=[];saved_folds=[]
with warnings.catch_warnings(record=True) as recorded,threadpool_limits(limits=1):
 warnings.simplefilter('always')
 for seed in [100,101,102]:
  for fold,(it,iv) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y,groups)):
   assert not set(groups[it])&set(groups[iv])
   model=clone(pipeline).fit(X[it],y[it]);pred=model.predict(X[iv]);score=model.decision_function(X[iv])
   folds.append({'seed':seed,'fold':fold,'balanced_accuracy':float(balanced_accuracy_score(y[iv],pred))})
   saved_folds.append({'seed':seed,'fold':fold,'train_ids':tr.iloc[it].track_id.tolist(),'validation_ids':tr.iloc[iv].track_id.tolist()})
   f=OUT/f'fold_{seed}_{fold}.joblib';joblib.dump(model,f);assert np.array_equal(joblib.load(f).predict(X[iv]),pred)
   for j,k in enumerate(iv):predrows.append({'seed':seed,'fold':fold,'track_id':tr.iloc[k].track_id,'label':int(y[k]),'prediction':int(pred[j]),'score':float(score[j])})
 final=clone(pipeline).fit(X,y);xt=te[cols].to_numpy(float);yp=final.predict(xt);ys=final.decision_function(xt)
 joblib.dump(final,OUT/'reproduced_candidate.joblib');assert np.array_equal(joblib.load(OUT/'reproduced_candidate.joblib').predict(xt),yp)
 measured=metrics(te.label.to_numpy(),yp,ys)
 ci=artist_bootstrap(te.label.to_numpy(),yp,ys,te.artist.map(normalize_artist).to_numpy())
 warning_rows=[{'category':w.category.__name__,'message':str(w.message)} for w in recorded]
foldtable=pd.DataFrame(folds);foldtable.to_csv(OUT/'fold_scores.csv',index=False)
pd.DataFrame(predrows).to_csv(OUT/'development_predictions.csv',index=False)
h=te[['track_id','artist','title','label']].copy();h['prediction']=yp;h['score']=ys;h.to_csv(OUT/'historical_predictions.csv',index=False)
dump(OUT/'folds.json',saved_folds);dump(OUT/'warnings.json',warning_rows)
claimed=json.loads((INCOMING/'results_v2/summary.json').read_text())
assert np.isclose(foldtable.balanced_accuracy.mean(),claimed['selected_cv_bacc'],atol=1e-12)
assert all(np.isclose(measured[k],v,atol=1e-12) for k,v in claimed['holdout_metrics'].items())
assert all(np.allclose(ci[k],v,atol=1e-12) for k,v in claimed['holdout_ci95_artist_bootstrap'].items())
source_sweep=pd.read_csv(INCOMING/'results_v2/config_sweep.csv')
null=pd.read_csv(INCOMING/'results_v2/null_best_of_n.csv')
pilot=pd.read_csv(ARCHIVE/'pilot_manifest.csv')
pilot_meta={'rows':len(pilot),'artist_names':int(pilot.artist.nunique()),'status_counts':pilot.chart_status.value_counts().to_dict(),'audio_files_present':sum((INCOMING/p).exists() for p in pilot.audio_path),'top_level_manifest_identical_to_nested':sha(ARCHIVE/'pilot_manifest.csv')==sha(INCOMING/'data/pilot_manifest.csv'),'label_and_album_verification':'Not independently verified; source script describes memory-based candidates and chart-history file is absent'}
existing=load_historical(ROOT/'results/evaluations/v2_nested_001_historical')
result={'status':'complete','reproduction_scope':'Fixed winner: 15 grouped CV fits plus one historical fit. Full 65-setting search and four permutation sweeps not rerun.','completed_fits':16,'failed_fits':0,'warnings_count':len(warning_rows),'elapsed_seconds':round(time.time()-started,2),'train_rows':len(tr),'test_rows':len(te),'feature_count':len(cols),'cv_mean_balanced_accuracy':float(foldtable.balanced_accuracy.mean()),'cv_sd_folds':float(foldtable.balanced_accuracy.std(ddof=1)),'historical_metrics':measured,'historical_confusion_matrix':confusion_matrix(te.label,yp).tolist(),'historical_ci95':ci,'claims_reproduced':True,'existing_v2_historical_metrics':existing['metrics'],'archive_minus_existing_v2_historical_ba_points':100*(measured['balanced_accuracy']-existing['metrics']['balanced_accuracy']),'archive_sweep_rows':len(source_sweep),'archive_null_sweeps':len(null),'pilot':pilot_meta,'new_audio_files':0,'incoming_deployed_model_identical_to_original_baseline':True,'raw_archive_sha256':sha(Path('/Users/dhanushss/Downloads/files (1).zip')),'no_fresh_test':True}
dump(OUT/'summary.json',result)
manifest={'status':'complete','run_id':'archive_review_001','python':platform.python_version(),'versions':{k:version(k) for k in ['numpy','pandas','scikit-learn','joblib','threadpoolctl']},'inputs':inputs,'archive_source_files':{str(p.relative_to(INCOMING)):sha(p) for p in INCOMING.rglob('*') if p.is_file()},'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
dump(OUT/'manifest.json',manifest)
print(json.dumps(result,indent=2))
