"""Small, cached public metadata readers. No audio downloading or paid services."""
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from html.parser import HTMLParser
import json
import time
import pandas as pd
from .common import digest, sha, write, read, stamp
from .schema import require

UT_COMMIT = '24084586b74e6c4d40f95e90cf9a57f1b6388331'
YEAR_COMMIT = 'edb3d94c0517abbb023604c71ed9ca4549465965'
UA = 'ChorusHitPredictorAcademic/1.0 (https://github.com/DhanushSS/chorus-hit-predictor)'
PUBLIC_FILES = {
    'weekly_current.csv': f'https://raw.githubusercontent.com/utdata/rwd-billboard-data/{UT_COMMIT}/data-out/hot-100-current.csv',
    'LICENSE_UTDATA.txt': f'https://raw.githubusercontent.com/utdata/rwd-billboard-data/{UT_COMMIT}/LICENSE',
    'utdata_notes.md': f'https://raw.githubusercontent.com/utdata/rwd-billboard-data/{UT_COMMIT}/index.qmd',
    'yearend.csv': f'https://raw.githubusercontent.com/Patrick5225/Billboard-Year-End-Top-Songs/{YEAR_COMMIT}/wikipedia_scraper.csv',
    'LICENSE_YEAREND.txt': f'https://raw.githubusercontent.com/Patrick5225/Billboard-Year-End-Top-Songs/{YEAR_COMMIT}/LICENSE',
    'yearend_2021.html': 'https://en.wikipedia.org/wiki/Special:PermanentLink/1362868665',
}


class PublicCache:
    def __init__(self, directory):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.last_mb = 0.

    def fetch(self, url, filename=None):
        require(filename is None or Path(filename).name == filename, 'Cache filename must not contain a path')
        path = self.directory / (filename or (digest(url) + '.json'))
        provenance = path.with_suffix(path.suffix + '.source.json')
        if path.exists() and provenance.exists():
            note = read(provenance)
            require(note['url'] == url and note['sha256'] == sha(path), 'Public source cache hash/URL mismatch')
            return path
        if path.exists():
            # An unprovenanced pre-existing download is never silently trusted.
            raise ValueError(f'Cached file lacks provenance: {path}; use a new cache directory')
        if url.startswith('https://musicbrainz.org/ws/2/'):
            time.sleep(max(0, 1.1 - (time.monotonic() - self.last_mb)))
        require(url.startswith(('https://musicbrainz.org/ws/2/', 'https://raw.githubusercontent.com/', 'https://en.wikipedia.org/wiki/')), 'Only declared public metadata hosts are supported')
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={'User-Agent': UA}), timeout=40) as response:
                    data = response.read(30 * 1024 * 1024 + 1)
                require(len(data) <= 30 * 1024 * 1024, 'Metadata response exceeds size bound')
                break
            except HTTPError as error:
                if error.code not in {429, 502, 503, 504} or attempt == 2:
                    raise
                time.sleep(min(30, max(2 ** (attempt + 1), int(error.headers.get('Retry-After', '0')))))
            finally:
                if url.startswith('https://musicbrainz.org/'):
                    self.last_mb = time.monotonic()
        with path.open('xb') as output:
            output.write(data)
        write(provenance, {'url': url, 'sha256': sha(path), 'retrieved_at': stamp(), 'kind': 'public_metadata_not_audio'})
        return path

    def mb(self, entity, **params):
        url = 'https://musicbrainz.org/ws/2/' + entity + '?' + urlencode({'fmt': 'json', **params})
        path = self.fetch(url)
        return read(path), {'url': url, 'sha256': sha(path), 'cache_file': path.name, 'license': 'MusicBrainz core metadata CC0; no audio rights implied'}


class WikiRows(HTMLParser):
    """Extract only numeric rank/title/artist rows; do not execute page content."""
    def __init__(self):
        super().__init__(); self.rows = []; self.row = []; self.cell = None; self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row = []
        if tag in {'td', 'th'}: self.cell = []
        if tag == 'sup': self.skip += 1

    def handle_endtag(self, tag):
        if tag == 'sup': self.skip = max(0, self.skip - 1)
        if tag in {'td', 'th'} and self.cell is not None:
            self.row.append(''.join(self.cell).strip()); self.cell = None
        if tag == 'tr' and len(self.row) >= 3 and self.row[0].isdigit():
            self.rows.append(self.row[:3])

    def handle_data(self, data):
        if self.cell is not None and not self.skip: self.cell.append(data)


def read_public_tables(directory):
    root = Path(directory)
    weekly = pd.read_csv(root / 'weekly_current.csv').rename(columns={'chart_week': 'date', 'current_week': 'rank', 'performer': 'artist'})
    require({'date','rank','title','artist'} <= set(weekly.columns), 'Unexpected weekly CSV format')
    annual = pd.read_csv(root / 'yearend.csv', encoding='cp1252')
    parser = WikiRows(); parser.feed((root / 'yearend_2021.html').read_text())
    extra = pd.DataFrame([{'rank': int(rank), 'title': title.strip('"'), 'artist': artist, 'year': 2021} for rank, title, artist in parser.rows])
    require(not extra.empty, '2021 year-end page has no usable ranking')
    require(set(extra['rank']) == set(range(1, 101)) and len(extra) == 100, '2021 year-end page is not a full unique ranking')
    annual = pd.concat([annual, extra], ignore_index=True)
    annual = annual[annual.year.between(2006, 2021)].copy()
    ranks = pd.to_numeric(annual['rank'], errors='raise')
    require((ranks % 1 == 0).all(), 'Annual ranks must be integers')
    annual['rank'] = ranks.astype(int)
    for year, g in annual.groupby('year'):
        require(len(g) == 100 and set(g['rank']) == set(range(1, 101)), f'Incomplete year-end ranking: {year}')
    require(set(annual.year) == set(range(2006, 2022)), 'Missing year-end year')
    return weekly[['date','rank','title','artist']], annual[['year','rank','title','artist']]
