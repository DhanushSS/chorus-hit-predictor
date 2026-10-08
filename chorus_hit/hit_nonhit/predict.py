"""Predict with a separately evaluated genuine new-target model on authorized local audio."""
import json
from .artifacts import load, predict_features
from .audio import contract, extract
from .common import cli, parser


def main():
    p = parser(__doc__); p.add_argument('--run', required=True); p.add_argument('--audio', required=True)
    p.add_argument('--start', type=float)
    a = p.parse_args(); b, _ = load(a.run, require_evaluated=True)
    if a.dry_run:
        print(json.dumps({'task_id': b['task_id'], 'chart_cutoff': b['chart_cutoff'], 'audio_decoded': False})); return
    X, _, meta = extract(a.audio, a.start)
    print(json.dumps({**predict_features(b, X, contract()), 'excerpt': meta, 'uploaded_audio_accuracy': 'Not measured for this upload'}, indent=2))


if __name__ == '__main__':
    cli(main)
