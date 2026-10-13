"""Public CSV contract, tested before the immutable prediction boundary."""
import numpy as np
import pandas as pd
import pytest
from chorus_hit.config import ROOT
from chorus_hit.hsp_s.evaluate import load_model, predict_frame
from chorus_hit.ui_components.csv_input import parse_csv, MAX_CSV_BYTES

@pytest.fixture(scope='module')
def sample():
    return pd.read_csv(ROOT/'data/example_features.csv')

def encoded(frame): return frame.to_csv(index=False).encode()

def test_valid_csv_preserves_actual_saved_predictions(sample):
    parsed = parse_csv(encoded(sample), list(sample.columns))
    assert predict_frame(ROOT/'results/hsp_s/run_001', parsed) == predict_frame(ROOT/'results/hsp_s/run_001', sample)

@pytest.mark.parametrize('kind, message', [('missing','Missing 1'),('extra','Extra 1'),('duplicate','Duplicate'),('order','wrong order')])
def test_schema_errors(sample, kind, message):
    f=sample.copy()
    if kind=='missing': f=f.iloc[:,1:]
    if kind=='extra': f['peakPos']=1
    if kind=='duplicate': f.columns=[f.columns[0]]+list(f.columns[:-1])
    if kind=='order': f=f.iloc[:,::-1]
    with pytest.raises(ValueError,match=message): parse_csv(encoded(f),list(sample.columns))

def test_reorder_is_explicit_and_lossless(sample):
    pd.testing.assert_frame_equal(parse_csv(encoded(sample.iloc[:,::-1]),list(sample.columns),reorder=True),sample)

@pytest.mark.parametrize('value,message',[('oops','record 1'),('inf','infinity'),('-inf','infinity'),('9'*129,'128 characters')])
def test_bad_cell(sample,value,message):
    f=sample.astype(object); f.iloc[0,0]=value
    with pytest.raises(ValueError,match=message): parse_csv(encoded(f),list(sample.columns))

def test_missing_cells_compatible_but_empty_record_rejected(sample):
    f=sample.copy(); f.iloc[0,0]=np.nan
    assert np.isnan(parse_csv(encoded(f),list(sample.columns)).iloc[0,0])
    f.iloc[0,:]=np.nan
    with pytest.raises(ValueError,match='all feature values'): parse_csv(encoded(f),list(sample.columns))

@pytest.mark.parametrize('payload,message',[(b'','empty'),(b'\xff','UTF-8'),(b'\x00','binary'),(b'"unclosed','parsed'),(b'x'*(MAX_CSV_BYTES+1),'5 MB')])
def test_bad_bytes(payload,message):
    with pytest.raises(ValueError,match=message): parse_csv(payload,['a'])

def test_record_and_legacy_bounds():
    for payload,message in [(b'a\n','no records'),(b'a\n'+b'1\n'*101,'100 records'),(b'a\n1,2\n','expected 1'),((','.join('c'+str(i) for i in range(518))+'\n').encode(),'legacy 518')]:
        with pytest.raises(ValueError,match=message): parse_csv(payload,['a'])
    assert len(parse_csv(b'a\n'+b'1\n'*100,['a']))==100
