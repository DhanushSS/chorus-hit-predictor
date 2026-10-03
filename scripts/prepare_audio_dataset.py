"""Extract permitted, verified new recordings with the SAME path used by uploads."""
import argparse,json,sys,tempfile,os
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chorus_hit.ingestion import validate_record,connected_groups
from chorus_hit.audio import extract_record
from chorus_hit.audit import atomic_json,sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); records=json.loads(a.manifest.read_text())['records']
    if not records: p.error('Manifest contains no recordings')
    for r in records: validate_record(r,require_audio=True)
    if len({r['recording_id'] for r in records})!=len(records): p.error('One row per recording required; segments must aggregate before scoring')
    versions={r['dataset_version'] for r in records}; extractors={r['extractor_version'] for r in records}
    if len(versions)!=1 or len(extractors)!=1: p.error('One dataset and extractor version per collection')
    if a.out.exists(): p.error('Output exists; choose a new versioned directory')
    groups=connected_groups(records); rows=[]; metadata=[]
    for r in records:
        X,segment,meta=extract_record(r['audio_path_or_authorized_reference'],r['segment_start_seconds'],r['extractor_version'])
        rows.append({'track_id':r['recording_id'],'artist':r['original_artist_credit'],'title':r['title'],'label':r['label'],**X.iloc[0].to_dict()})
        metadata.append({**r,**meta,'artist_group_id':groups[r['recording_id']]})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent,prefix='.audio-') as tmp:
        d=Path(tmp)/'dataset';d.mkdir();pd.DataFrame(rows).to_csv(d/'chorus_features.csv',index=False)
        atomic_json(d/'manifest.json',{'dataset_version':next(iter(versions)),'extractor_version':next(iter(extractors)),
            'data_sha256':sha256_file(d/'chorus_features.csv'),'records':metadata,'evaluation_status':'not_trained'})
        os.rename(d,a.out)
    print(a.out)

if __name__=='__main__':main()
