"""Install the checked-in, checksum-pinned model bundle without retraining."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install(root=ROOT):
    from chorus_hit.hsp_s.evaluate import load_model
    root = Path(root)
    receipt = json.loads((root / 'models/bundle.json').read_text())
    archive = root / 'models/hsp_s_run_001.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('Model archive checksum mismatch')
    destination = root / 'results/hsp_s/run_001'
    if destination.exists():
        if hashlib.sha256((destination / 'manifest.json').read_bytes()).hexdigest() != receipt['model_manifest_sha256']:
            raise ValueError('Existing model differs; it will not be overwritten')
        load_model(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.install-', dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as z:
            entries = z.infolist()
            if len(entries) != receipt['files'] or sum(e.file_size for e in entries) > 512 * 1024 * 1024:
                raise ValueError('Unexpected archive contents or size')
            if len({e.filename for e in entries}) != len(entries):
                raise ValueError('Duplicate archive entry')
            for entry in entries:
                name = PurePosixPath(entry.filename)
                if name.is_absolute() or '..' in name.parts or '\\' in entry.filename or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError('Unsafe archive path')
                target = temporary / name
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(entry) as source, target.open('xb') as output:
                        shutil.copyfileobj(source, output)
        if hashlib.sha256((temporary / 'manifest.json').read_bytes()).hexdigest() != receipt['model_manifest_sha256']:
            raise ValueError('Model manifest mismatch')
        load_model(temporary)  # Validate every file, code identity and environment before installation.
        temporary.rename(destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == '__main__':
    print('Model ready:', install().relative_to(ROOT))
