from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from chorus_hit.config import FEATURE_COLUMNS, ROOT
from chorus_hit.hit_nonhit import EXTRACTOR
from chorus_hit.hit_nonhit.artifacts import load, predict_features
from chorus_hit.hit_nonhit.audio import contract
from chorus_hit.hit_nonhit.audit import assert_isolated, connected_groups
from chorus_hit.hit_nonhit.build_dataset import load_dataset
from chorus_hit.hit_nonhit.common import digest, read, sha, verify
from chorus_hit.hit_nonhit.evaluate_locked import evaluate
from chorus_hit.hit_nonhit.freeze_split import freeze, grouped_folds, load_development
from chorus_hit.hit_nonhit.modeling import estimator
from chorus_hit.hit_nonhit.synthetic import feature_package, inputs
from chorus_hit.hit_nonhit.train import train
from chorus_hit.task_artifacts import load_for_task


@pytest.fixture(scope='module')
def package(tmp_path_factory):
    root = tmp_path_factory.mktemp('new_task')
    feature_package(root / 'dataset'); freeze(root / 'dataset', root / 'split')
    return root


def test_dataset_evidence_and_fold_isolation(package):
    X, rows, m, cfg = load_dataset(package / 'dataset')
    dev, sm, c = load_development(package / 'split')
    ids = read(package / 'split/split.json')
    assert set(ids['development']).isdisjoint(ids['locked'])
    assert_isolated(rows, ids['development'], ids['locked'])
    for train_i, valid_i in grouped_folds(dev.label.to_numpy(), dev.group.to_numpy(), 3, 20261008):
        assert set(dev.group.iloc[train_i]).isdisjoint(dev.group.iloc[valid_i])
    assert list(X.columns) == ['recording_id'] + FEATURE_COLUMNS


@pytest.mark.parametrize('key,value', [('canonical_artist_ids',['same_artist']),('canonical_work_id','same_work'),('album_id','same_album'),('match_set_id','same_match'),('audio_sha256','same_audio'),('feature_row_hash','same_features'),('perceptual_duplicate_group','same_reviewed_cluster')])
def test_shared_identity_or_audio_cannot_cross(key, value):
    rows, _, _ = inputs(2)
    rows = [rows[0], rows[2]]
    for r in rows: r[key] = value
    assert len(set(connected_groups(rows))) == 1
    with pytest.raises(ValueError, match='leakage'): assert_isolated(rows, [rows[0]['recording_id']], [rows[1]['recording_id']])


def test_collaboration_components_transitive():
    rows, _, _ = inputs(3); rows = [rows[0], rows[2], rows[4]]
    rows[0]['canonical_artist_ids'] = ['a']; rows[1]['canonical_artist_ids'] = ['a','b']; rows[2]['canonical_artist_ids'] = ['b']
    assert len(set(connected_groups(rows))) == 1


def test_too_few_groups_blocks_not_seed_search():
    with pytest.raises(ValueError, match='Insufficient'): grouped_folds(np.array([0,1,0,1]), np.array(['a','a','b','b']), 3, 42)


def test_fold_local_scaler_and_audio_only_columns():
    cfg = {'id':'lr','family':'logistic','representation':'std74','C':.1}
    X = pd.DataFrame(np.random.default_rng(5).normal(size=(20, 518)), columns=FEATURE_COLUMNS)
    X.loc[15:] += 1e5
    model = estimator(cfg, 42).fit(X.iloc[:15], np.arange(15) % 2)
    assert abs(model.named_steps['scale'].mean_).max() < 2
    assert len(model.named_steps['scale'].mean_) == 74
    assert list(model.feature_names_in_) == FEATURE_COLUMNS


def test_nested_training_task_guard_and_one_time_test(package):
    split, run = package / 'split', package / 'run'
    assert train(split, run, dry_run=True)['test_labels_loaded'] is False
    result = train(split, run)
    assert result['synthetic'] and result['test_accessed'] is False
    assert not (split / 'test_access.json').exists()
    with pytest.raises(ValueError, match='Synthetic'): load(run)
    b, m = load(run, allow_synthetic=True)
    dev, _, _ = load_development(split)
    out = predict_features(b, dev[FEATURE_COLUMNS], contract())
    assert len(out['predictions']) == len(dev) and not out['score_semantics']['calibrated']
    with pytest.raises(ValueError, match='feature order'): predict_features(b, dev[FEATURE_COLUMNS[::-1]], contract())
    with pytest.raises(ValueError, match='extractor'): predict_features(b, dev[FEATURE_COLUMNS], {'version':'legacy-librosa-518-v1'})
    with pytest.raises(ValueError, match='Completed'): train(split, run)
    dry = evaluate(split, run, dry_run=True, allow_synthetic=True)
    assert not (split / 'test_access.json').exists() and not dry['test_labels_loaded']
    report = evaluate(split, run, allow_synthetic=True)
    assert report['evaluation_status'] == 'synthetic_test_only'
    assert report['metrics']['confusion_matrix'] and (split / 'test_access.json').exists()
    with pytest.raises(ValueError, match='already accessed'): evaluate(split, run, allow_synthetic=True)
    with pytest.raises(ValueError, match='already accessed'): train(split, package / 'second_run')


def test_legacy_bundle_rejected_as_new_target():
    with pytest.raises(ValueError, match='Wrong task'): load(ROOT / 'results/v2/v4_std74_001')
    with pytest.raises(ValueError, match='Unknown task'): load_for_task('not_a_task')


def test_hash_tampering_and_membership_tampering(package, tmp_path):
    import shutil
    folder = tmp_path / 'split'; shutil.copytree(package / 'split', folder)
    (folder / 'development.csv').write_text('tampered')
    with pytest.raises(ValueError, match='hash'): load_development(folder)
    shutil.rmtree(folder); shutil.copytree(package / 'split', folder)
    data = read(folder / 'split.json'); data['development'][0] = data['locked'][0]
    (folder / 'split.json').write_text(json.dumps(data))
    m = read(folder / 'manifest.json'); m['files']['split.json'] = sha(folder / 'split.json'); m['split_hash'] = digest(data)
    (folder / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='membership|overlap'): load_development(folder)


def test_freeze_deterministic_and_no_overwrite(package, tmp_path):
    detail = freeze(package / 'dataset', tmp_path / 'second')
    assert detail == read(package / 'split/split.json')
    with pytest.raises(ValueError, match='already exists'): freeze(package / 'dataset', tmp_path / 'second')


@pytest.mark.parametrize('module', ['chart_membership','build_dataset','freeze_split','train','evaluate_locked','predict','smoke'])
def test_clis_have_real_help(module):
    result = subprocess.run([sys.executable,'-m','chorus_hit.hit_nonhit.' + module,'--help'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and '--dry-run' in result.stdout


def test_all_preserved_assets_unchanged():
    manifest = read(ROOT / 'docs/hit_nonhit/preservation_manifest.json')
    for name, expected in manifest['files'].items():
        path = ROOT / name
        # Ignored PDFs/decks are local; tracked dataset/model/config bytes are mandatory in CI.
        if name.startswith('output/') and not path.exists(): continue
        assert sha(path) == expected, name
