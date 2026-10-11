from io import BytesIO
import numpy as np
import soundfile as sf
import streamlit as st
from streamlit.testing.v1 import AppTest
from chorus_hit.config import ROOT

class Upload:
    def __init__(self,payload,name): self.payload=payload; self.name=name
    def getvalue(self): return self.payload


def test_csv_upload_buttons_output_and_errors(monkeypatch):
    current=Upload((ROOT/'data/example_features.csv').read_bytes(),'sample.csv')
    monkeypatch.setattr(st,'file_uploader',lambda *a,**kw:current if kw['key']=='csv_upload' else None)
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception
    assert any('Validated 4 rows × 436' in e.value for e in app.success)
    assert len(app.dataframe[-1].value)==4
    assert 'Decision margin' in app.dataframe[-1].value
    current.payload=b'bad,columns\n1,2\n'
    app.button[0].click().run()
    assert not app.exception and any('Schema mismatch' in e.value for e in app.error)


def test_mp3_ui_requires_permission_and_returns_preview_only(monkeypatch):
    buffer=BytesIO(); t=np.arange(22050)/22050
    sf.write(buffer,.2*np.sin(2*np.pi*220*t),22050,format='MP3')
    upload=Upload(buffer.getvalue(),'generated.mp3')
    monkeypatch.setattr(st,'file_uploader',lambda *a,**kw:upload if kw['key']=='mp3_upload' else None)
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert app.button[1].disabled
    app.checkbox[1].check().run(); app.button[1].click().run()
    assert not app.exception
    assert any(m.label=='Sample rate' and m.value=='22,050 Hz' for m in app.metric)
    assert any('No hit score was calculated' in e.value for e in app.info)
    assert not app.dataframe
