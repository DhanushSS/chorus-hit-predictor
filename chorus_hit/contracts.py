"""Semantic checks in addition to byte integrity; these do not establish provenance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ROOT, LABELS
from .audit import sha256_file, json_hash, TASK_VERSION, GROUP_VERSION, LEGACY_DATASET
from .data import normalize_artist
from .evaluation import classification_metrics, target_status, grouped_folds

FROZEN_SPLIT_SHA256 = 'e8be2658ec1cdd33c89009e72e987c2d139336d96efc306fa6065b34ae29b3e4'


def require_fields(value, fields, name):
    if not isinstance(value, dict) or set(fields) - set(value):
        raise ValueError(f'Missing required {name} fields: {sorted(set(fields)-set(value)) if isinstance(value,dict) else fields}')


def metrics_match(reported, calculated):
    require_fields(reported, calculated, 'metrics')
    for k, expected in calculated.items():
        actual = reported[k]
        equal = (isinstance(actual, (int, float)) and np.isclose(actual, expected, rtol=0, atol=1e-12)) if isinstance(expected, float) else actual == expected
        if not equal: raise ValueError(f'Summary metric mismatch: {k}')


def frozen_split(data, path=None, expected_sha256=FROZEN_SPLIT_SHA256):
    path = Path(path) if path is not None else ROOT/'results/split_manifest.csv'
    if sha256_file(path) != expected_sha256 or expected_sha256 != FROZEN_SPLIT_SHA256:
        raise ValueError('Frozen split manifest hash mismatch')
    split = pd.read_csv(path)
    validate_rows(split, data, data.track_id)
    if set(split.partition) != {'train','test'}: raise ValueError('Invalid split partitions')
    groups = data.set_index('track_id').artist.map(normalize_artist)
    dev = split.loc[split.partition=='train','track_id'].tolist()
    historical = split.loc[split.partition=='test','track_id'].tolist()
    if set(groups.loc[dev]) & set(groups.loc[historical]): raise ValueError('Historical artist overlap')
    return split, dev, historical


def validate_rows(rows, data, expected_ids, *, groups_required=False):
    needed = {'track_id','label'} | ({'artist_group'} if groups_required else set())
    if needed-set(rows.columns): raise ValueError('Missing evaluation columns')
    if rows.track_id.isna().any() or rows.track_id.duplicated().any() or set(rows.track_id)!=set(expected_ids):
        raise ValueError('Evaluation membership mismatch (expected exactly once)')
    actual = data.set_index('track_id').loc[rows.track_id]
    if not np.array_equal(actual.label, rows.label): raise ValueError('Prediction labels are misaligned')
    if 'artist_group' in rows and not np.array_equal(actual.artist.map(normalize_artist),rows.artist_group):
        raise ValueError('Prediction artist groups are misaligned')
    for col in ('artist','title'):
        if col in rows and not np.array_equal(actual[col], rows[col]): raise ValueError(f'Evaluation {col} mismatch')


def validate_fold(fold, data, population):
    require_fields(fold, ['fold','fit_track_ids','validation_track_ids'], 'fold')
    fit, val = fold['fit_track_ids'], fold['validation_track_ids']
    if len(set(fit))!=len(fit) or len(set(val))!=len(val) or set(fit)&set(val) or set(fit)|set(val)!=set(population):
        raise ValueError('Invalid fold membership')
    names = data.set_index('track_id').artist.map(normalize_artist)
    if set(names.loc[fit]) & set(names.loc[val]): raise ValueError('Fold artist overlap')


def validate_evaluation(path, manifest, summary, bundle, data, predictions, trials):
    require_fields(summary, ['evaluation_status','metrics','target_status','counts','dataset_rows','task_version','group_version'], 'summary')
    if manifest['task_version']!=TASK_VERSION or manifest['group_version']!=GROUP_VERSION or manifest['dataset']['dataset_version']!=LEGACY_DATASET:
        raise ValueError('Unsupported task/dataset/group contract')
    if manifest['labels']!={str(k):v for k,v in LABELS.items()}: raise ValueError('Label meaning mismatch')
    _, dev, historical = frozen_split(data)
    status = summary['evaluation_status']
    if set(bundle['train_track_ids']) != set(dev): raise ValueError('Final training membership mismatch')
    if manifest.get('evaluation_status', 'historical_test') != status:
        raise ValueError('Evaluation status mismatch')
    if status not in {'historical_test','nested_development','tuning_development'}: raise ValueError('Unsupported evaluation status')
    if summary.get('fresh_test_available',False): raise ValueError('Legacy data is not a fresh test')
    expected = historical if status=='historical_test' else dev
    validate_rows(predictions, data, expected, groups_required=status!='historical_test')
    if not {'prediction','score'}.issubset(predictions): raise ValueError('Missing prediction columns')
    if not np.isfinite(predictions.score).all(): raise ValueError('Nonfinite evaluation score')
    metrics_match(summary['metrics'], classification_metrics(predictions.label,predictions.prediction,
                   None if status=='nested_development' else predictions.score))
    if summary['target_status'] != target_status(summary['metrics'],status): raise ValueError('Target check mismatch')
    if summary['dataset_rows']!=len(data): raise ValueError('Dataset row count mismatch')
    counts = summary['counts']
    expected_counts={'train_songs':len(dev),'train_artists':data.loc[data.track_id.isin(dev)].artist.map(normalize_artist).nunique()}
    expected_counts.update({'test_songs':len(historical),'test_artists':data.loc[data.track_id.isin(historical)].artist.map(normalize_artist).nunique()} if status=='historical_test' else
                          {'evaluation_songs':len(dev),'evaluation_artists':predictions.artist_group.nunique(),'historical_songs_reserved':len(historical)})
    for k,v in expected_counts.items():
        if counts.get(k)!=v: raise ValueError(f'Count mismatch: {k}')
    if status=='historical_test':
        if set(manifest.get('evaluation_track_ids',[]))!=set(historical): raise ValueError('Historical membership mismatch')
        return
    for name in ('config.json','membership.csv','final_development_folds.json','final_development_ranking.csv'):
        if name not in manifest['files']: raise ValueError(f'Unverified required evaluation asset: {name}')
    config=json.loads((path/'config.json').read_text())
    if json_hash(config)!=manifest['config_sha256']: raise ValueError('Config identity mismatch')
    if (config['mode']=='nested') != (status=='nested_development'): raise ValueError('Config evaluation status mismatch')
    membership=pd.read_csv(path/'membership.csv'); validate_rows(membership,data,data.track_id)
    expected_parts=pd.Series(['development' if v in set(dev) else 'historical_test' for v in membership.track_id])
    if not np.array_equal(membership.partition,expected_parts): raise ValueError('Development membership mismatch')
    if len({r['key'] for r in trials})!=len(trials): raise ValueError('Duplicated trial records')
    bykey={r['key']:r for r in trials}
    def winner(stage, population):
        for name in (f'{stage}_ranking.csv',f'{stage}_folds.json'):
            if name not in manifest['files']: raise ValueError('Unverified selection asset')
        rank=pd.read_csv(path/f'{stage}_ranking.csv')
        specs={v['id']:v for v in config['candidates']}
        if set(rank.candidate_id)!=set(specs) or rank.candidate_id.duplicated().any(): raise ValueError('Ranking candidate membership mismatch')
        folds=json.loads((path/f'{stage}_folds.json').read_text())
        if len(folds)!=config['inner_folds']: raise ValueError('Inner fold count mismatch')
        subset=data.set_index('track_id').loc[population].reset_index()
        declared=grouped_folds(subset,config['inner_folds'],config['inner_seed'],subset.artist.map(normalize_artist),manifest['raw_schema'])
        expected_folds=[{'fold':i,'fit_track_ids':subset.track_id.iloc[f].tolist(),'validation_track_ids':subset.track_id.iloc[v].tolist()} for i,(f,v) in enumerate(declared)]
        if folds!=expected_folds: raise ValueError('Inner fold assignment mismatch')
        for f in folds: validate_fold(f,data,population)
        if sorted(v for f in folds for v in f['validation_track_ids'])!=sorted(population): raise ValueError('Inner coverage mismatch')
        for row in rank.itertuples():
            rs=[bykey.get(f'{stage}_{row.candidate_id}_{f["fold"]}') for f in folds]
            if any(r is None or r['spec']!=specs[row.candidate_id] for r in rs): raise ValueError('Ranking/trial spec mismatch')
            success=[r for r in rs if r['status']=='complete']
            eligible=len(success)==len(folds)
            if bool(row.eligible)!=eligible: raise ValueError('Ranking eligibility mismatch')
            if eligible and not np.isclose(row.mean_balanced_accuracy,np.mean([r['metrics']['balanced_accuracy'] for r in success]),rtol=0,atol=1e-12): raise ValueError('Ranking score mismatch')
        valid=rank[rank.eligible].sort_values(['mean_balanced_accuracy','candidate_id'],ascending=[False,True])
        if valid.empty: raise ValueError('No eligible winner')
        chosen=valid.iloc[0]
        # Check the actual inner-fold records for the selected candidate.
        for f in folds:
            r=bykey[f'{stage}_{chosen.candidate_id}_{f["fold"]}']; pred=pd.DataFrame(r['predictions'])
            validate_rows(pred,data,f['validation_track_ids'],groups_required=True)
            metrics_match(r['metrics'],classification_metrics(pred.label,pred.prediction,pred.score))
            if r['fold']!=f['fold'] or r['stage']!=stage: raise ValueError('Trial stage/fold mismatch')
            if r['fit_rows']!=len(f['fit_track_ids']) or r['validation_rows']!=len(pred): raise ValueError('Inner trial row count mismatch')
        return chosen
    final=winner('final_development',dev)
    if final.candidate_id!=summary['model_name']: raise ValueError('Final winner mismatch')
    spec=next(c for c in config['candidates'] if c['id']==final.candidate_id)
    if summary.get('candidate')!=spec or not np.isclose(summary['tuning_balanced_accuracy'],final.mean_balanced_accuracy,rtol=0,atol=1e-12): raise ValueError('Final candidate/selection mismatch')
    if status=='nested_development':
        for name in ('outer_folds.json','outer_fold_metrics.json'):
            if name not in manifest['files']: raise ValueError('Unverified outer fold asset')
        folds=json.loads((path/'outer_folds.json').read_text()); outer=json.loads((path/'outer_fold_metrics.json').read_text())
        development=data.loc[data.track_id.isin(dev)].reset_index(drop=True)
        declared=grouped_folds(development,config['outer_folds'],config['outer_seed'],development.artist.map(normalize_artist),manifest['raw_schema'])
        expected_folds=[{'fold':i,'fit_track_ids':development.track_id.iloc[fit].tolist(),'validation_track_ids':development.track_id.iloc[val].tolist()} for i,(fit,val) in enumerate(declared)]
        if folds!=expected_folds or len(outer)!=len(folds) or set(predictions.fold)!=set(range(len(folds))): raise ValueError('Outer fold assignment mismatch')
        for f, reported in zip(folds,outer):
            validate_fold(f,data,dev); selected=winner(f'outer{f["fold"]}',f['fit_track_ids'])
            pred=predictions[predictions.fold==f['fold']]
            validate_rows(pred,data,f['validation_track_ids'],groups_required=True)
            if reported['fold']!=f['fold'] or reported['candidate_id']!=selected.candidate_id or set(pred.candidate_id)!={selected.candidate_id}: raise ValueError('Outer winner mismatch')
            trial=bykey.get(f'outer{f["fold"]}_assessment_{selected.candidate_id}')
            if trial is None or trial['status']!='complete' or trial['fit_rows']!=len(f['fit_track_ids']) or trial['validation_rows']!=len(pred): raise ValueError('Outer assessment trial mismatch')
            records=pd.DataFrame(trial['predictions']).set_index('track_id').sort_index()
            observed=pred.set_index('track_id').sort_index()
            try: pd.testing.assert_frame_equal(records,observed[records.columns],check_dtype=False,rtol=0,atol=1e-12)
            except AssertionError as exc: raise ValueError('Outer predictions/trial mismatch') from exc
            calculated=classification_metrics(pred.label,pred.prediction,pred.score)
            metrics_match(reported,calculated);metrics_match(trial['metrics'],calculated)
    else:
        folds=json.loads((path/'final_development_folds.json').read_text())
        if set(predictions.fold)!=set(f['fold'] for f in folds): raise ValueError('Tuning fold mismatch')
        for f in folds:
            pred=predictions[predictions.fold==f['fold']]
            validate_rows(pred,data,f['validation_track_ids'],groups_required=True)
            if set(pred.candidate_id)!={final.candidate_id}: raise ValueError('Tuning winner mismatch')
