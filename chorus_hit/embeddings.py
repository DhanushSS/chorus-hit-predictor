"""Optional local-only frozen MERT adapter and content-addressed embedding cache.

No weights or remote code are downloaded automatically. Real experiments require
permitted recordings, a pinned local model and a completed code/weight review.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import os
import tempfile

import librosa
import numpy as np
from .audit import json_hash,sha256_file,atomic_json


def embedding_key(audio_sha256,segment,spec):
    required={'model_id','revision','sample_rate','layer','pooling','processor_config','extractor_version'}
    if not required.issubset(spec): raise ValueError(f'Embedding spec needs {sorted(required)}')
    if spec['pooling']!='mean_time': raise ValueError('Only predeclared mean-time pooling is supported')
    if len(spec['revision'])!=40 or any(c not in '0123456789abcdef' for c in spec['revision']): raise ValueError('Pin a full immutable model revision')
    if not isinstance(spec['sample_rate'],int) or spec['sample_rate']<=0: raise ValueError('Invalid model sampling rate')
    return json_hash({'audio_sha256':audio_sha256,'segment':segment,'spec':spec})


def cached_embedding(audio_path,start_seconds,duration_seconds,spec,cache_dir,encoder):
    path=Path(audio_path)
    if not path.is_file(): raise FileNotFoundError('Real local audio is required for embeddings')
    if not np.isfinite([start_seconds,duration_seconds]).all() or start_seconds<0 or duration_seconds<=0:
        raise ValueError('Invalid embedding segment')
    segment={'start_seconds':start_seconds,'duration_seconds':duration_seconds}
    key=embedding_key(sha256_file(path),segment,spec); cache=Path(cache_dir); cache.mkdir(parents=True,exist_ok=True)
    target=cache/f'{key}.npz'; meta=cache/f'{key}.json'
    if target.exists() and meta.exists():
        identity=json.loads(meta.read_text())
        if identity['array_sha256']!=sha256_file(target): raise ValueError('Embedding cache hash mismatch')
        with np.load(target,allow_pickle=False) as z: vector=z['embedding']
        if vector.ndim!=1 or not np.isfinite(vector).all(): raise ValueError('Invalid cached embedding')
        return vector,key
    y,sr=librosa.load(path,sr=spec['sample_rate'],mono=True,offset=start_seconds,duration=duration_seconds)
    if len(y)!=round(duration_seconds*sr) or not np.isfinite(y).all() or np.sqrt(np.mean(y.astype(float)**2))<1e-5:
        raise ValueError('Embedding needs a complete finite non-silent segment')
    vector=np.asarray(encoder(y,sr,spec),dtype=np.float32)
    if vector.ndim!=1 or not len(vector) or not np.isfinite(vector).all(): raise ValueError('Encoder must return one finite vector per recording segment')
    fd,tmp=tempfile.mkstemp(dir=cache,suffix='.npz'); os.close(fd)
    try:
        np.savez_compressed(tmp,embedding=vector); os.replace(tmp,target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    atomic_json(meta,{'key':key,'audio_sha256':sha256_file(path),'segment':segment,'spec':spec,
                     'dimensions':len(vector),'array_sha256':sha256_file(target)})
    return vector,key


class LocalMERTEncoder:
    def __init__(self,model_dir,review_path):
        self.path=Path(model_dir); self.review=json.loads(Path(review_path).read_text())
        if self.review.get('review_status')!='approved_local_code' or not self.review.get('intended_use_license_compatible'):
            raise ValueError('Local code review and license compatibility record required')
        files=self.review.get('files',{})
        if not files: raise ValueError('Reviewed model/processor/code hashes required')
        for name,digest in files.items():
            path=(self.path/name).resolve()
            if not path.is_relative_to(self.path.resolve()) or sha256_file(path)!=digest: raise ValueError('Local model review hash mismatch')
        present={str(p.relative_to(self.path)) for p in self.path.rglob('*') if p.is_file()}
        sensitive={p for p in present if p.endswith(('.py','.json','.safetensors','.bin'))}
        if not sensitive.issubset(files): raise ValueError('Unreviewed local model files')
        import torch
        from transformers import Wav2Vec2FeatureExtractor,AutoModel
        self.torch=torch
        self.processor=Wav2Vec2FeatureExtractor.from_pretrained(self.path,local_files_only=True)
        self.model=AutoModel.from_pretrained(self.path,local_files_only=True,trust_remote_code=True).to('cpu').eval()

    def __call__(self,y,sr,spec):
        if spec['revision']!=self.review['revision'] or sr!=self.processor.sampling_rate:
            raise ValueError('Pinned model/processor sampling-rate contract mismatch')
        if spec['processor_config']!=json.loads((self.path/'preprocessor_config.json').read_text()):
            raise ValueError('Processor configuration mismatch')
        inputs=self.processor(y,sampling_rate=sr,return_tensors='pt')
        with self.torch.inference_mode():
            output=self.model(**inputs,output_hidden_states=True)
            return output.hidden_states[spec['layer']].mean(dim=1).squeeze(0).cpu().numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--start',type=float,default=0);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args()
    if not a.audio.is_file(): p.error('Audio missing; no model is loaded')
    spec=json.loads(a.spec.read_text()); encoder=LocalMERTEncoder(a.model_dir,a.review)
    v,key=cached_embedding(a.audio,a.start,15,spec,a.cache,encoder);print(json.dumps({'key':key,'dimensions':len(v),'runtime':{x:version(x) for x in ['torch','transformers']}}))

if __name__=='__main__':main()
