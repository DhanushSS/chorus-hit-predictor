"""Immutable local artifacts. Hashes detect changes; they are not source attestation."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from chorus_hit.config import ROOT
from . import TASK


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Exclusive, atomic JSON publication; never replace an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.link(name, path)  # atomic no-clobber, including concurrent invocations
    finally:
        os.unlink(name)


@contextmanager
def stage(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / ('.' + destination.name + '.lock')
    with lock.open('x'):
        pass
    tmp = Path(tempfile.mkdtemp(prefix='.building-', dir=destination.parent))
    try:
        yield tmp
        if destination.exists():
            raise ValueError('Destination appeared during build')
        tmp.rename(destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
        lock.unlink()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def environment():
    return {p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'librosa', 'joblib', 'soundfile']}


def runtime_check():
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Python 3.12 is required')
    if sys.platform == 'win32':
        raise ValueError('Training is supported on macOS/Linux only')
    pins = dict(line.strip().split('==') for line in (ROOT / 'requirements.txt').read_text().splitlines() if '==' in line)
    for name, actual in environment().items():
        if pins.get(name) != actual:
            raise ValueError(f'Pinned environment mismatch: {name}')


def code_identity():
    paths = sorted((ROOT / 'chorus_hit' / 'hit_nonhit').glob('*.py')) + [ROOT / 'chorus_hit/audio.py', ROOT / 'chorus_hit/config.py']
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'implementation': {str(p.relative_to(ROOT)): sha(p) for p in paths}}


def seal(folder, metadata):
    folder = Path(folder)
    files = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in {'manifest.json', '.training.lock'}}
    manifest = {'task_id': TASK, **metadata, 'files': files}
    write(folder / 'manifest.json', manifest)
    return manifest


def verify(folder, kind=None):
    folder = Path(folder)
    m = read(folder / 'manifest.json')
    if m.get('task_id') != TASK or (kind and m.get('kind') != kind):
        raise ValueError('Wrong task or artifact kind')
    require_files = {'dataset': {'records.json', 'features.csv', 'candidates.json', 'archive.json', 'config.json'},
                     'split': {'development.csv', 'locked.csv', 'identity.json', 'split.json', 'config.json', 'development_records.json'},
                     'model': {'model.joblib', 'summary.json', 'selection.json', 'config.json', 'diagnostics.json'}}
    if not require_files.get(m.get('kind'), set()) <= set(m['files']):
        raise ValueError('Manifest omits required artifact files')
    permitted_extra = {'manifest.json'} | ({'test_access.json'} if m.get('kind') == 'split' else {'locked_evaluation.json'} if m.get('kind') == 'model' else set())
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    if actual - permitted_extra != set(m['files']):
        raise ValueError('Unmanifested or missing artifact files')
    for name, expected in m['files'].items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder.resolve()) or not p.is_file() or sha(p) != expected:
            raise ValueError(f'Artifact hash/path mismatch: {name}')
    return m


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--dry-run', action='store_true', help='Validate inputs and print status without writing or fitting')
    return p


def cli(function):
    try:
        function()
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f'Blocked: {error}', file=sys.stderr)
        raise SystemExit(2) from error
