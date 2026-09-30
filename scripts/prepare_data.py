"""Rebuild the bundled feature table from the pinned, licensed source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chorus_hit.config import DATA_PATH, FEATURE_COLUMNS, SOURCE_COMMIT, SOURCE_REPO

URL = f"https://raw.githubusercontent.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus/{SOURCE_COMMIT}/Devolp/Final%20Data.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="Optional local copy of the original CSV")
    args = parser.parse_args()
    if args.source:
        raw = args.source.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            raw = response.read()
    import io
    source = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    assert source.shape == (751, 523), "Pinned source schema changed"
    assert source.columns[5:].tolist() == FEATURE_COLUMNS, "Unexpected source feature order"
    frame = source[["Artist", "Title", "Label", *FEATURE_COLUMNS]].rename(
        columns={"Artist": "artist", "Title": "title", "Label": "label"})
    frame.insert(0, "track_id", [f"CH{i:04d}" for i in range(1, len(frame)+1)])
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    provenance = {
        "source_repository": SOURCE_REPO, "source_commit": SOURCE_COMMIT,
        "source_file": "Devolp/Final Data.csv", "source_url": URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "prepared_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "license": "Apache-2.0, retained in licenses/DATASET_APACHE_2_0.txt",
        "rows": 751, "audio_features": 518,
        "changes": ["Removed source audio paths", "Renamed Artist/Title/Label", "Added stable track_id"],
        "label_1": "Year-end Hot-100-Songs entries collected by upstream notebook for 2006-2021",
        "label_0": "Weekly hot-100 entries from sampled dates in 2006-2021, excluding titles already collected",
        "critical_limitation": "Both labels can be charting songs. This is a year-end-hit proxy, not never-charted versus charted. Labels have not been independently re-audited against every chart.",
        "feature_provenance": "Upstream states pychorus 15-second extraction followed by librosa statistics. Original audio and the exact library environment are unavailable here.",
    }
    (DATA_PATH.parent / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Prepared {len(frame)} songs with {len(FEATURE_COLUMNS)} audio features")


if __name__ == "__main__":
    main()
