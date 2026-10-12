"""Prevent broken landing-page assets and accidental scientific relabelling."""
import hashlib
import json
import re
from urllib.parse import unquote
from chorus_hit.config import ROOT


def test_readme_relative_links_and_actual_images_exist():
    text=(ROOT/'README.md').read_text()
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if target.startswith(('https://','http://','#')):continue
        path=ROOT/unquote(target.split('#')[0])
        assert path.exists(),target
    for name in ['app_overview.jpg','app_csv.jpg','app_mp3_preview.jpg']:
        content=(ROOT/'docs/assets'/name).read_bytes()
        assert content.startswith(b'\xff\xd8\xff') and len(content)>10000
    assert '# Billboard Hit Prediction Using Audio Features' in text
    assert 'PES2UG24CS154' in text and 'PES2UG24CS150' in text
    assert '80% was not reached' in text and 'new untouched external test is NOT AVAILABLE' in text


def test_all_24_issue_ids_reported_once():
    text=(ROOT/'docs/audit/issue_resolution_matrix.md').read_text()
    ids=re.findall(r'\| (ISSUE-\d{3}) \|',text)
    assert ids==[f'ISSUE-{i:03d}' for i in range(1,25)]


def test_published_baseline_hash_receipt_and_research_boundary():
    for name,sha in json.loads((ROOT/'docs/audit/baseline_preservation.json').read_text()).items():
        if name.startswith('output/'):continue # private split is not required in CI
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    result=json.loads((ROOT/'docs/research_v2/results.json').read_text())
    assert result['external_test']=='NOT_AVAILABLE' and result['promotion']=='NONE'
    assert result['identity']['old_test_loaded'] is False
    assert len(result['trials'])==226 and all(t['status']=='complete' for t in result['trials'])
    assert result['nested']['confusion_matrix']==[[2099,985],[780,2371]]
