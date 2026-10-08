"""Human-reviewed canonical identity decisions. Text similarity never creates labels."""
import unicodedata
from .common import digest, read, write
from .schema import require


def normalize(text):
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in unicodedata.normalize('NFKC', text).casefold()).split())


def append_decision(directory, decision):
    from pathlib import Path
    for k in ['recording_id', 'reviewer', 'reviewed_at', 'evidence_urls', 'resolution', 'reason']:
        require(bool(decision.get(k)), f'Identity decision missing {k}')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / '.decision.lock'
    with lock.open('x'):
        pass
    try:
        entries = sorted(directory.glob('*.json'))
        previous = digest(read(entries[-1])) if entries else None
        item = {**decision, 'previous_sha256': previous}
        write(directory / f'{len(entries):06d}-{digest(item)[:12]}.json', item)
        return item
    finally:
        lock.unlink()



def review_queue(labels):
    return sorted([{'recording_id': x['recording_id'], 'reason': x['excluded_reason']} for x in labels if x['label'] is None], key=lambda x: x['recording_id'])
