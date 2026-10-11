"""Generated tones exercise decoding; they are NOT labelled music evaluations."""
from pathlib import Path
import io
import numpy as np
import pytest
import soundfile as sf
from chorus_hit.audio_upload.service import preview_mp3, MAX_AUDIO_BYTES
from chorus_hit.audio_upload import worker

@pytest.fixture
def mp3():
    buffer=io.BytesIO(); t=np.arange(22050)/22050
    sf.write(buffer,.2*np.sin(2*np.pi*440*t),22050,format='MP3')
    return buffer.getvalue()

def test_real_mp3_decode_preview_and_cleanup(mp3,tmp_path):
    r=preview_mp3(mp3,'generated.mp3',temporary_parent=tmp_path)
    assert r['status']=='PREVIEW_ONLY' and r['prediction'] is None
    assert r['duration_seconds']==pytest.approx(1,abs=.15)
    assert r['sample_rate']==22050 and r['rms']>0 and r['waveform_envelope']
    assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('payload,name,message',[(b'','a.mp3','empty'),(b'x','a.wav','MP3 file'),(b'x'*(MAX_AUDIO_BYTES+1),'a.mp3','25 MB'),(b'garbage','a.mp3','corrupt, truncated or unsupported')])
def test_invalid_audio_and_cleanup(payload,name,message,tmp_path):
    with pytest.raises(ValueError,match=message): preview_mp3(payload,name,temporary_parent=tmp_path)
    assert list(tmp_path.iterdir())==[]

def test_extension_spoof_and_silence_rejected(tmp_path):
    for fmt,message in [('WAV','not an MP3'),('MP3','silent')]:
        b=io.BytesIO(); sf.write(b,np.zeros(22050),22050,format=fmt)
        with pytest.raises(ValueError,match=message): preview_mp3(b.getvalue(),'fake.mp3',temporary_parent=tmp_path)
        assert list(tmp_path.iterdir())==[]

def test_real_timeout_removes_temporary_data(mp3,tmp_path):
    with pytest.raises(ValueError,match='timed out'): preview_mp3(mp3,'a.mp3',timeout=.00001,temporary_parent=tmp_path)
    assert list(tmp_path.iterdir())==[]

def test_truncated_mp3_rejected(mp3,tmp_path):
    with pytest.raises(ValueError): preview_mp3(mp3[:100],'truncated.mp3',temporary_parent=tmp_path)
    assert list(tmp_path.iterdir())==[]

class FakeStream:
    format='MP3'; samplerate=22050; channels=1; frames=22050
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def blocks(self,**kwargs): yield np.ones((22050,1),dtype='float32')

@pytest.mark.parametrize('field,value,message',[('samplerate',96000,'8–48'),('channels',6,'mono/stereo'),('frames',22050*481,'duration')])
def test_header_limits_without_large_allocation(monkeypatch,field,value,message):
    stream=FakeStream(); setattr(stream,field,value); monkeypatch.setattr(worker.sf,'SoundFile',lambda _:stream)
    with pytest.raises(ValueError,match=message): worker.analyze('mock-header')

def test_declared_frame_truncation_and_stream_overrun(monkeypatch):
    s=FakeStream(); s.frames=50000
    monkeypatch.setattr(worker.sf,'SoundFile',lambda _:s)
    with pytest.raises(ValueError,match='truncated'): worker.analyze('mock-header')
    s.frames=1; s.samplerate=8000
    s.blocks=lambda **kwargs: (np.ones((16384,1),dtype='float32') for _ in range(240))
    with pytest.raises(ValueError,match='duration/size'): worker.analyze('mock-header')

def test_decoder_crash_cleanup(monkeypatch,mp3,tmp_path):
    import chorus_hit.audio_upload.service as service
    class Failed:
        pid=99999999
        def wait(self,**kwargs): return 1
        def poll(self): return 1
    monkeypatch.setattr(service.subprocess,'Popen',lambda *a,**k:Failed())
    monkeypatch.setattr(service,'stop_process_group',lambda p:None)
    with pytest.raises(ValueError,match='decoder failed'): preview_mp3(mp3,'a.mp3',temporary_parent=tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_opposite_phase_stereo_is_not_silence(tmp_path):
    b=io.BytesIO(); wave=.2*np.sin(2*np.pi*440*np.arange(22050)/22050)
    sf.write(b,np.column_stack([wave,-wave]),22050,format='MP3')
    assert preview_mp3(b.getvalue(),'stereo.mp3',temporary_parent=tmp_path)['rms']>.01
