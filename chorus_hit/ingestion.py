"""Versioned recording contracts and evidence-backed connected identity groups."""
from datetime import date
import json
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
        if record['label'] not in (0,1) or record['label_evidence_status']!='verified': raise ValueError('Verified label evidence required for new audio study')
        if not record['canonical_artist_ids'] or not record.get('identity_evidence_source'): raise ValueError('Canonical identity and evidence are required')
        if not record['audio_rights_basis']: raise ValueError('Audio rights basis required')
        path=Path(record['audio_path_or_authorized_reference'] or '')
        if not path.is_file(): raise FileNotFoundError(f'Missing local recording: {record["recording_id"]}')
        if sha256_file(path)!=record['audio_sha256']: raise ValueError('Audio hash mismatch')
        if record['segment_duration_seconds']!=15: raise ValueError('Declared audio study requires 15-second segments')
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
        if not artists or not r.get('identity_evidence_source'): raise ValueError('Unresolved artist identity blocks strict grouping')
        nodes=['artist:'+x for x in artists]+['recording:'+r['recording_id']]
        if r.get('duplicate_group_id'):
            if not r.get('duplicate_evidence_source'): raise ValueError('Duplicate grouping needs evidence')
            nodes.append('duplicate:'+r['duplicate_group_id'])
        for n in nodes: union(nodes[0],n)
    members={}
    for r in records: members.setdefault(find('recording:'+r['recording_id']),[]).append(r['recording_id'])
    return {rid:'GROUP-'+json_hash(sorted(ids))[:20] for ids in members.values() for rid in ids}


def coverage_report(records):
    rows=[]
    for r in records:
        reasons=[]
        for key in ['canonical_artist_ids','audio_path_or_authorized_reference','audio_sha256','audio_rights_basis','recording_version']:
            if not r.get(key): reasons.append(key)
        if r.get('label_evidence_status')!='verified': reasons.append('verified_label_evidence')
        rows.append({'recording_id':r['recording_id'],'artist':r['original_artist_credit'],'label':r['label'],
                     'eligible':not reasons,'missing':reasons})
    return rows
