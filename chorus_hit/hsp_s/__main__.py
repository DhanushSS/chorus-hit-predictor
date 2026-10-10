"""Separate HSP-S benchmark: download, prepare, train and evaluate once."""
import argparse
import json
from chorus_hit.hit_nonhit.common import cli


def main():
    p = argparse.ArgumentParser(__doc__); s = p.add_subparsers(dest='command', required=True)
    d = s.add_parser('download'); d.add_argument('--out', required=True)
    d = s.add_parser('prepare'); d.add_argument('--source', required=True); d.add_argument('--out', required=True); d.add_argument('--config', default='configs/hsp_s_acousticbrainz.json')
    d = s.add_parser('train'); d.add_argument('--split', required=True); d.add_argument('--out', required=True); d.add_argument('--resume', action='store_true')
    d = s.add_parser('evaluate'); d.add_argument('--split', required=True); d.add_argument('--run', required=True)
    a = p.parse_args()
    if a.command == 'download':
        from .download import download
        result = str(download(a.out))
    elif a.command == 'prepare':
        from .data import prepare
        result = prepare(a.source, a.out, a.config)
    elif a.command == 'train':
        from .train import train
        result = train(a.split, a.out, a.resume)
    else:
        from .evaluate import evaluate
        result = evaluate(a.split, a.run)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': cli(main)
