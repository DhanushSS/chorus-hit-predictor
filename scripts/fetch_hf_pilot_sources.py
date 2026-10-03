"""Restore pinned public pilot inputs without credentials or paid services."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/hf_feasibility_001'


def main():
    manifest = json.loads((OUT / 'source_manifest.json').read_text())
    for name, spec in manifest['sources'].items():
        # The Spotify metadata inspection is not needed to reproduce the fit.
        if name == 'music4all_metadata.csv':
            continue
        destination = OUT / name
        if destination.exists():
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == spec['sha256'], name
            print(f'Verified existing {name}')
            continue
        response = requests.get(spec['url'], timeout=(20, 90))
        response.raise_for_status()
        assert len(response.content) == spec['bytes'], name
        assert hashlib.sha256(response.content).hexdigest() == spec['sha256'], name
        destination.write_bytes(response.content)
        print(f'Downloaded and verified {name}')


if __name__ == '__main__':
    main()
