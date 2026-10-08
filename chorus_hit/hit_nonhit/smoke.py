"""Bounded synthetic-only end-to-end check. Does not report research accuracy."""
import json
from .common import cli, parser
from .synthetic import feature_package
from .freeze_split import freeze
from .train import train
from .evaluate_locked import evaluate
from pathlib import Path


def main():
    p = parser(__doc__); p.add_argument('--out', required=True); a = p.parse_args()
    if a.dry_run:
        print('Synthetic test only: generated chart evidence -> synthetic features -> split -> fit -> one evaluation'); return
    root = Path(a.out)
    feature_package(root / 'dataset'); freeze(root / 'dataset', root / 'split')
    train(root / 'split', root / 'run'); evaluate(root / 'split', root / 'run', allow_synthetic=True)
    print(json.dumps({'status': 'passed', 'synthetic': True, 'research_accuracy': None, 'real_data_training_status': 'blocked_data'}))


if __name__ == '__main__':
    cli(main)
