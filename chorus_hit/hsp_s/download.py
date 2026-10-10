"""Resumable bounded downloads of the CC-BY-4.0 HSP-S feature release; no audio."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import requests
from . import SOURCE_URL, SOURCE_MD5, SOURCE_SIZE


def verify_source(path):
    path = Path(path)
    if path.stat().st_size != SOURCE_SIZE:
        raise ValueError('Incomplete HSP-S feature file')
    h = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    if h.hexdigest() != SOURCE_MD5:
        raise ValueError('HSP-S publisher checksum mismatch')


def download(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'hsp-s_acousticbrainz.parquet'
    if target.exists():
        verify_source(target); return target
    chunk = 2 * 1024 * 1024
    count = (SOURCE_SIZE + chunk - 1) // chunk
    def fetch(i):
        start = i * chunk; end = min(SOURCE_SIZE, start + chunk) - 1
        path = folder / f'part_{i:03d}'
        if path.exists() and path.stat().st_size == end - start + 1: return i
        for attempt in range(4):
            try:
                with requests.get(SOURCE_URL, headers={'Range': f'bytes={start}-{end}'}, timeout=(15, 30)) as response:
                    if response.status_code == 429: raise RuntimeError('Source rate limited; stop and retry later')
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SOURCE_SIZE}':
                        raise ValueError('Unexpected range response')
                    if len(response.content) != end - start + 1: raise ValueError('Truncated source chunk')
                    temporary = path.with_suffix('.partial'); temporary.write_bytes(response.content); temporary.replace(path)
                    return i
            except (requests.RequestException, ValueError):
                if attempt == 3: raise
                time.sleep(2 * (attempt + 1))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for done, future in enumerate(as_completed([pool.submit(fetch, i) for i in range(count)]), 1):
            future.result()
            if done % 10 == 0: print(f'Downloaded {done}/{count} feature chunks', flush=True)
    temporary = target.with_suffix('.assembling')
    with temporary.open('wb') as f:
        for i in range(count): f.write((folder / f'part_{i:03d}').read_bytes())
    verify_source(temporary); temporary.replace(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--out', required=True)
    print(download(parser.parse_args().out))
