import json
import numpy as np
import pytest
from chorus_hit.ingestion import connected_groups,stable_recording_id,validate_record,FIELDS
from chorus_hit.embeddings import cached_embedding,embedding_key
from chorus_hit.audio import extract_features,extraction_config,load_audio,select_segment,extract_record
from chorus_hit.config import SAMPLE_RATE


def test_identity_connections_do_not_split_names():
    records=[{'recording_id':'a','canonical_artist_ids':['artist:earth-wind-and-fire'],'identity_evidence_source':'fixture'},
             {'recording_id':'b','canonical_artist_ids':['artist:earth-wind-and-fire','guest:x'],'identity_evidence_source':'fixture'},
             {'recording_id':'c','canonical_artist_ids':['guest:x'],'identity_evidence_source':'fixture'}]
    assert len(set(connected_groups(records).values()))==1
    with pytest.raises(ValueError,match='Unresolved'): connected_groups([{'recording_id':'u','original_artist_credit':'A & B'}])
    assert stable_recording_id('v2','recording:a')==stable_recording_id('v2','recording:a')
    assert stable_recording_id('v3','recording:a')!=stable_recording_id('v2','recording:a')


def test_shared_audio_and_invalid_inputs(tmp_path):
    import soundfile as sf
    sr=24000;t=np.arange(15*sr)/sr;y=.1*np.sin(2*np.pi*220*t)+.06*np.sin(2*np.pi*330*t)
    path=tmp_path/'tone.wav';sf.write(path,y,sr)
    X,segment,meta=extract_record(path,0,'shared-librosa-518-v2')
    assert X.shape==(1,518);assert meta['sample_rate']==SAMPLE_RATE
    assert meta['extractor_config']['zcr_frame_length']==2048
    assert extraction_config()['zcr_frame_length']==22050
    assert np.isfinite(extract_features(y,sr).to_numpy()).all()
    with pytest.raises(ValueError,match='finite'): extract_features(np.full(15*SAMPLE_RATE,np.nan))
    with pytest.raises(ValueError,match='Unknown'): extract_features(segment.audio,extractor_version='unknown')
    bad=tmp_path/'broken.wav';bad.write_bytes(b'not an audio file')
    with pytest.raises(Exception): load_audio(bad)
    over=tmp_path/'over.wav';sf.write(over,np.ones(481*1000)*.1,1000)
    with pytest.raises(ValueError,match='longer'): load_audio(over)


def test_embedding_cache_and_sampling_rate(tmp_path):
    import soundfile as sf
    sr=16000;t=np.arange(sr)/sr;audio=tmp_path/'tone.wav';sf.write(audio,.1*np.sin(2*np.pi*220*t),sr)
    spec={'model_id':'software-fixture','revision':'a'*40,'sample_rate':24000,'layer':-1,'pooling':'mean_time','processor_config':{'sampling_rate':24000},'extractor_version':'test'}
    calls=[]
    def fake_encoder(y,sr,spec):
        calls.append(sr);assert sr==24000 and len(y)==sr
        return np.array([y.mean(),y.std(),y.max()])
    v,key=cached_embedding(audio,0,1,spec,tmp_path/'cache',fake_encoder)
    w,key2=cached_embedding(audio,0,1,spec,tmp_path/'cache',fake_encoder)
    assert len(calls)==1 and key==key2 and np.array_equal(v,w)
    assert embedding_key('x',{'start':0},{**spec,'layer':-2})!=embedding_key('x',{'start':0},spec)
    assert embedding_key('x',{'start':1},spec)!=embedding_key('x',{'start':0},spec)
    with pytest.raises(FileNotFoundError): cached_embedding(tmp_path/'absent',0,1,spec,tmp_path/'cache',fake_encoder)
    with pytest.raises(ValueError,match='complete'): cached_embedding(audio,0,2,spec,tmp_path/'cache',fake_encoder)
    with pytest.raises(ValueError,match='revision'): embedding_key('x',{}, {**spec,'revision':'main'})


@pytest.mark.parametrize('start',[-1,-1e-9,float('nan'),float('inf'),1.01])
def test_manual_boundaries(start):
    y=.1*np.sin(2*np.pi*220*np.arange(16*SAMPLE_RATE)/SAMPLE_RATE)
    with pytest.raises(ValueError):select_segment(y,start_seconds=start)


def test_silence_short_and_centered_fallback():
    with pytest.raises(ValueError,match='silent'):select_segment(np.zeros(15*SAMPLE_RATE))
    with pytest.raises(ValueError,match='15 seconds'):select_segment(np.ones(14*SAMPLE_RATE)*.1)
    y=.1*np.sin(2*np.pi*220*np.arange(16*SAMPLE_RATE)/SAMPLE_RATE)
    s=select_segment(y);assert 'Centered excerpt' in s.method and s.start_seconds==.5
    assert select_segment(y,start_seconds=1).start_seconds==1


@pytest.mark.parametrize('extension,format,subtype',[('wav','WAV','PCM_16'),('flac','FLAC','PCM_16'),('ogg','OGG','VORBIS'),('mp3','MP3','MPEG_LAYER_III')])
def test_supported_upload_codecs(tmp_path,extension,format,subtype):
    import soundfile as sf
    # Generated tones test decoding, never musical predictive validity.
    assert format in sf.available_formats(),f'Pinned environment missing advertised {format} codec'
    sr=22050;y=.1*np.sin(2*np.pi*220*np.arange(16*sr)/sr)
    p=tmp_path/f'tone.{extension}';sf.write(p,y,sr,format=format,subtype=subtype)
    decoded,rate=load_audio(p);assert rate==SAMPLE_RATE and len(decoded)>=15*rate


@pytest.mark.parametrize('frames,method',[(100,'few candidates'),(260,'no separated repeat'),(600,'Automatic repeated-segment candidate')])
def test_selector_fallback_metadata(monkeypatch,frames,method):
    import chorus_hit.audio as module
    chroma=np.tile(np.sin(np.arange(frames)*.1),(12,1))
    monkeypatch.setattr(module.librosa.feature,'chroma_stft',lambda **kwargs:chroma)
    segment=select_segment(np.ones(60*SAMPLE_RATE)*.1)
    assert method in segment.method
    if method.startswith('Automatic'):assert np.isfinite(segment.repetition_similarity)
    else:assert segment.repetition_similarity is None
