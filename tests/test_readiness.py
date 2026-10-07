import json
from chorus_hit.readiness import evidence_audit,check_readiness
from chorus_hit.config import ROOT


def test_current_readiness_and_evidence_counts():
    r=check_readiness()
    assert r['software_ready'],r['errors']
    assert r['active_run']=='v1_baseline' and len(r['runs'])==4
    e=r['evidence'];assert e['records']==e['inherited']==e['unresolved_recordings']==751
    assert e['verified_with_references']==e['strict_audio_study_eligible']==e['disputed']==0
    assert e['waveform_parity_verified'] is False


def test_package_mismatch_is_reported(monkeypatch):
    import chorus_hit.readiness as module
    actual=module.version
    monkeypatch.setattr(module,'version',lambda p:'wrong' if p=='numpy' else actual(p))
    r=module.check_readiness();assert not r['software_ready']
    assert any('numpy: requires' in e for e in r['errors'])


def test_saved_evidence_registry_and_approval_match_sources():
    from chorus_hit.audit import sha256_file
    assert json.loads((ROOT/'docs/evidence_audit.json').read_text())==evidence_audit()
    registry=json.loads((ROOT/'docs/evaluation_registry.json').read_text())
    assert registry['dataset_sha256']==sha256_file(ROOT/'data/chorus_features.csv')
    assert len(registry['memberships']['development'])==597 and len(registry['memberships']['historical'])==154
    assert registry['current_fresh_test_available'] is False
    assert 'approved_as_reported_by_user' in (ROOT/'docs/task_alignment.md').read_text()


def test_readiness_without_site_packages_returns_report():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-S','-m','chorus_hit.readiness'],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==1
    r=json.loads(result.stdout)
    assert not r['software_ready'] and r['packages']['numpy']['installed'] is None
    assert any('dependencies unavailable' in e for e in r['errors'])
