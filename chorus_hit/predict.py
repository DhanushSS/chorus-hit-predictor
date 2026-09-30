"""Command-line inference for a held-out record or local audio."""
import argparse
import json
import joblib

from .config import FEATURE_COLUMNS, LABELS, MODEL_PATH
from .data import load_data
from .train import model_scores


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--track-id", help="A bundled track ID, e.g. CH0001")
    source.add_argument("--audio", help="Path to a local audio file")
    parser.add_argument("--start", type=float, help="Manual chorus start in seconds")
    args = parser.parse_args()
    bundle = joblib.load(MODEL_PATH)
    if args.audio:
        from .audio import extract_features, load_audio, select_segment
        y, sr = load_audio(args.audio)
        segment = select_segment(y, sr, args.start)
        X = extract_features(segment.audio, sr)
        info = {"start_seconds": segment.start_seconds, "selection": segment.method,
                "experimental_audio": True}
    else:
        frame = load_data()
        row = frame.loc[frame.track_id == args.track_id]
        if row.empty:
            parser.error("Unknown track ID")
        X = row[FEATURE_COLUMNS]
        info = {"track_id": args.track_id, "title": row.title.iloc[0], "artist": row.artist.iloc[0],
                "actual_label": int(row.label.iloc[0]),
                "held_out": args.track_id not in bundle["train_track_ids"]}
    predicted = int(bundle["pipeline"].predict(X)[0])
    score, score_type = model_scores(bundle["pipeline"], X)
    print(json.dumps({**info, "model": bundle["model_name"], "prediction": predicted,
                      "class_name": LABELS[predicted], "score": float(score[0]),
                      "score_type": score_type}, indent=2))


if __name__ == "__main__":
    main()
