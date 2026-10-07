"""Versioned recording contracts and evidence-backed connected identity groups."""
from datetime import date
import json
import math
import re
from urllib.parse import urlparse
from pathlib import Path
from .audit import json_hash,sha256_file

FIELDS=('recording_id','dataset_version','title','canonical_artist_ids','original_artist_credit',
        'release_date','recording_version','label','label_definition_version','label_evidence_status',
        'label_source','chart_market','observation_start','observation_end','audio_path_or_authorized_reference',
        'audio_sha256','audio_rights_basis','extractor_version','segment_start_seconds','segment_duration_seconds',
        'artist_group_id','duplicate_group_id')


def stable_recording_id(dataset_version,canonical_recording_id):
    if not dataset_version or not canonical_recording_id: raise ValueError('Version and canonical recording identity required')
    return 'REC-'+json_hash([dataset_version,canonical_recording_id])[:24]


def validate_record(record, *, require_audio=False,forecasting=False):
    missing=set(FIELDS)-set(record)
    if missing: raise ValueError(f'Missing ingestion fields: {sorted(missing)}')
    if not record['recording_id'] or not record['dataset_version'] or not record['label_definition_version']:
        raise ValueError('Missing version or identity')
    if record['label'] not in (None,0,1): raise ValueError('Invalid binary label')
    if record['canonical_artist_ids'] is not None and not isinstance(record['canonical_artist_ids'],list):
        raise ValueError('Canonical artist IDs must be a list or null, never split an artist credit by punctuation')
    if require_audio:
        report = record_validation(record)
        if not report['eligible_for_audio_study']:
            raise ValueError('Invalid audio-study record: ' + '; '.join(report['reasons']))
    if forecasting:
        required=['release_date','observation_start','observation_end','prediction_cutoff','chart_market']
        if any(not record.get(k) for k in required): raise ValueError('Forecasting requires release, cutoff, market and observation window')
        dates={k:date.fromisoformat(record[k]) for k in required if k!='chart_market'}
        if not (dates['prediction_cutoff']<=dates['observation_start']<=dates['observation_end']): raise ValueError('Invalid prediction/outcome window')
        if dates['observation_end']>=date.today(): raise ValueError('Observation window has not finished')
    return record


def connected_groups(records):
    """Strict linked-performer/recording groups. Caller supplies evidenced IDs."""
    parent={}
    def find(a):
        parent.setdefault(a,a)
        if parent[a]!=a: parent[a]=find(parent[a])
        return parent[a]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for r in records:
        artists=r.get('canonical_artist_ids')
        if not _canonical_ids(artists) or not r.get('identity_evidence_source'): raise ValueError('Unresolved artist identity blocks strict grouping')
        nodes=['artist:'+x for x in artists]+['recording:'+r['recording_id']]
        if r.get('duplicate_group_id'):
            if not r.get('duplicate_evidence_source'): raise ValueError('Duplicate grouping needs evidence')
            nodes.append('duplicate:'+r['duplicate_group_id'])
        for n in nodes: union(nodes[0],n)
    members={}
    for r in records: members.setdefault(find('recording:'+r['recording_id']),[]).append(r['recording_id'])
    return {rid:'GROUP-'+json_hash(sorted(ids))[:20] for ids in members.values() for rid in ids}


def _reference(value):
    # A syntactically usable reference is not proof it was independently reviewed.
    return isinstance(value, str) and bool(urlparse(value).netloc) and urlparse(value).scheme in {'http', 'https'}


def _canonical_ids(value):
    return isinstance(value, list) and bool(value) and all(
        isinstance(v, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*:[^\s:]+', v) for v in value) and len(set(value)) == len(value)


def record_validation(record):
    """One structural eligibility contract; evidence authenticity requires human review."""
    r = record
    metadata, identity, audio = [], [], []
    missing = set(FIELDS) - set(r)
    if missing: metadata.append('missing fields: ' + ', '.join(sorted(missing)))
    for key in ('recording_id','dataset_version','label_definition_version','title',
                'original_artist_credit','recording_version','audio_rights_basis','chart_market','chart_type'):
        if not isinstance(r.get(key), str) or not r[key].strip(): metadata.append(key)
    if isinstance(r.get('label'), bool) or r.get('label') not in (0, 1): metadata.append('binary label required')
    if r.get('label_evidence_status') != 'verified': metadata.append('verified label evidence required')
    if not _reference(r.get('label_source')): metadata.append('label_source reference required')
    try:
        first, last = (date.fromisoformat(r[k]) for k in ('observation_start','observation_end'))
        if first > last: raise ValueError()
    except (KeyError, TypeError, ValueError): metadata.append('valid chart observation window required')
    if not _canonical_ids(r.get('canonical_artist_ids')): identity.append('canonical artist IDs must be unique namespace:ID strings')
    if not _reference(r.get('identity_evidence_source')): identity.append('identity_evidence_source reference required')
    if r.get('duplicate_group_id') and not _reference(r.get('duplicate_evidence_source')):
        identity.append('duplicate_evidence_source reference required')
    if r.get('extractor_version') not in {'legacy-librosa-518-v1','shared-librosa-518-v2'}:
        metadata.append('incompatible extractor_version')
    for key in ('segment_start_seconds','segment_duration_seconds'):
        v = r.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            metadata.append('invalid ' + key)
    if r.get('segment_duration_seconds') != 15: metadata.append('segment duration must be 15 seconds')
    path_value = r.get('audio_path_or_authorized_reference')
    path = Path(path_value) if isinstance(path_value, str) and path_value else None
    digest = r.get('audio_sha256')
    if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest): audio.append('invalid audio SHA-256')
    if path is None or not path.is_file(): audio.append('local audio file unavailable')
    elif sha256_file(path) != digest: audio.append('audio hash mismatch')
    else:
        try:
            import soundfile as sf
            info=sf.info(path)
            start=r.get('segment_start_seconds')
            if isinstance(start,(int,float)) and math.isfinite(start) and start+15 > info.duration + 1/info.samplerate:
                audio.append('segment extends outside recording')
        except (RuntimeError, ValueError, OSError): audio.append('audio container cannot be decoded')
    reasons = metadata + identity + audio
    return {'metadata_complete': not metadata, 'audio_available': not audio,
            'identity_verified': not identity, 'eligible_for_audio_study': not reasons,
            'reasons': reasons, 'verification_scope': 'structural checks against supplied evidence references; authenticity not independently verified'}


def coverage_report(records):
    rows=[]
    for r in records:
        result = record_validation(r)
        rows.append({'recording_id':r.get('recording_id'),'artist':r.get('original_artist_credit'),
                     'label':r.get('label'), **result,
                     'eligible':result['eligible_for_audio_study'],'missing':result['reasons']})
    return rows
