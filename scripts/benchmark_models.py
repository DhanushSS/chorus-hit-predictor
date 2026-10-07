"""Benchmark fixed model pipelines on identical artist-separated folds."""
import io
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from threadpoolctl import threadpool_limits

from chorus_hit.artifacts import load_run
from chorus_hit.data import normalize_artist
from chorus_hit.estimators import make_estimator
from chorus_hit.evaluation import grouped_folds

ROOT = Path(__file__).resolve().parents[1]
from chorus_hit.contracts import frozen_split
from chorus_hit.audit import sha256_file, atomic_json
from chorus_hit.diagnostics import identity, new_output, bounded_command
from chorus_hit.evaluation import positive_integer
import argparse
import sys


def benchmark(destination, config_path=ROOT/'configs/v4_std74.json', *, prediction_repeats=100, bootstrap_repeats=2000, _reserved=False):
    positive_integer(prediction_repeats,'prediction_repeats'); positive_integer(bootstrap_repeats,'bootstrap_repeats')
    destination=Path(destination) if _reserved else new_output(destination)
    config = json.loads(Path(config_path).read_text())
    specs = {s["id"]: s for s in config["candidates"]}
    names = ["v4_lr_std74_standard_0.1", "control_mlp", "control_random_forest"]
    run = load_run("v4_std74_001")
    split, _, _ = frozen_split(run.data, ROOT/config['split_manifest'],config['split_sha256'])
    if run.manifest['dataset']['source_sha256']!=config['data_sha256']: raise ValueError('Dataset/config hash mismatch')
    membership = split.set_index("track_id").loc[run.data.track_id]
    dev = run.data.loc[membership.partition.to_numpy() == "train"].reset_index(drop=True)
    batch = dev[run.bundle["features"]].iloc[:154]
    features = run.bundle["features"]
    folds = grouped_folds(dev, 5, 44, dev.artist.map(normalize_artist), features)
    output = {
        "status": "exploratory_matched_benchmark",
        "protocol": "five identical artist-separated folds; same 597 development songs; one CPU thread",
        "dataset": "year-end versus other weekly charted songs, not Eric Liu's task",
        "machine": platform.platform(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "results": {},
    }
    output['identity']=identity(__file__, status='exploratory_fixed_pipeline_development',
        config_sha256=sha256_file(config_path),split_sha256=config['split_sha256'],fold_seed=44,
        prediction_repeats=prediction_repeats,bootstrap_repeats=bootstrap_repeats,workers=1,
        timing_scope='fit and prediction only; extraction excluded',
        folds=[{'fit':dev.track_id.iloc[f].tolist(),'validation':dev.track_id.iloc[v].tolist()} for f,v in folds])
    atomic_json(destination/'protocol.json',output['identity'])
    rows = []
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for name in names:
            spec = specs[name]
            predictions = np.full(len(dev), -1, dtype=int)
            fit_seconds = []
            for fold, (fit, val) in enumerate(folds):
                model = make_estimator(spec, features, 42)
                start = time.perf_counter()
                model.fit(dev.iloc[fit][features], dev.iloc[fit].label)
                fit_seconds.append(time.perf_counter() - start)
                predictions[val] = model.predict(dev.iloc[val][features])
                for index in val:
                    rows.append({"track_id": dev.track_id.iloc[index], "artist_group": normalize_artist(dev.artist.iloc[index]),
                        "label": int(dev.label.iloc[index]), "model": name, "fold": fold, "prediction": int(predictions[index])})
                print(name, "fold", fold, "fit_seconds", round(fit_seconds[-1], 3), flush=True)
            assert (predictions >= 0).all()
            model = make_estimator(spec, features, 42)
            start = time.perf_counter()
            model.fit(dev[features], dev.label)
            full_fit_seconds = time.perf_counter() - start
            serialized = io.BytesIO()
            joblib.dump(model, serialized, compress=3)
            model.predict(batch)
            times = []
            for _ in range(prediction_repeats):
                start = time.perf_counter()
                model.predict(batch)
                times.append((time.perf_counter() - start) * 1000)
            output["results"][name] = {
                "accuracy": float(accuracy_score(dev.label, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(dev.label, predictions)),
                "precision": float(precision_score(dev.label, predictions, zero_division=0)),
                "recall": float(recall_score(dev.label, predictions)),
                "f1": float(f1_score(dev.label, predictions)),
                "fold_fit_seconds": fit_seconds,
                "median_fold_fit_seconds": float(np.median(fit_seconds)),
                "full_development_fit_seconds": full_fit_seconds,
                "median_predict_154_ms": float(np.median(times)),
                "compressed_pipeline_bytes": len(serialized.getvalue()),
                "prediction_songs": len(dev),
            }
        output["warnings"] = [str(w.message) for w in captured]
    saved = pd.DataFrame(rows)
    saved.to_csv(destination / "model_comparison_predictions.csv", index=False)
    wide = saved.pivot(index="track_id", columns="model", values="prediction")
    reference = saved.loc[saved.model == names[0]].set_index("track_id").loc[wide.index]
    y = reference.label.to_numpy()
    groups = reference.artist_group.to_numpy()
    unique_groups = np.unique(groups)
    baseline = wide[names[0]].to_numpy()

    def precision(labels, predictions):
        positive = predictions.sum()
        return float(((labels == 1) & (predictions == 1)).sum() / positive) if positive else 0.0

    output["paired_vs_controls"] = {}
    for rival_name in names[1:]:
        rival = wide[rival_name].to_numpy()
        random = np.random.default_rng(2026)
        samples = {"accuracy": [], "precision": []}
        for _ in range(bootstrap_repeats):
            indices = np.concatenate([np.flatnonzero(groups == group)
                                      for group in random.choice(unique_groups, len(unique_groups), replace=True)])
            samples["accuracy"].append(float(np.mean(baseline[indices] == y[indices])
                                             - np.mean(rival[indices] == y[indices])))
            samples["precision"].append(precision(y[indices], baseline[indices])
                                              - precision(y[indices], rival[indices]))
        output["paired_vs_controls"][rival_name] = {
            key: {"observed_difference": (float(np.mean(baseline == y) - np.mean(rival == y))
                                         if key == "accuracy" else precision(y, baseline) - precision(y, rival)),
                  "ci95": np.quantile(values, [0.025, 0.975]).tolist()}
            for key, values in samples.items()
        }
    (destination / "model_comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    for name, result in output["results"].items():
        print(f"{name}: accuracy={result['accuracy']:.1%}, "
              f"fit={result['median_fold_fit_seconds'] * 1000:.1f} ms/fold, "
              f"predict={result['median_predict_154_ms']:.1f} ms/154 songs, "
              f"size={result['compressed_pipeline_bytes'] / 1024:.1f} KiB")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout-seconds',type=float,default=120)
    p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a._worker: benchmark(a.out,_reserved=True)
    else: bounded_command([sys.executable,'-m','scripts.benchmark_models','--out',str(a.out.resolve()),'--_worker'],a.out,a.timeout_seconds)


if __name__=='__main__': main()
