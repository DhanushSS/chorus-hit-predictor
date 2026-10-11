"""Compare supplied known-recording scalar features; never enables app inference."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, extracted, columns, rtol=1e-4, atol=1e-6):
    from chorus_hit.audio_upload.feature_contract import map_features
    actual=np.array(map_features(extracted,columns)); expected=np.array(map_features(reference,columns))
    good=np.isclose(actual,expected,rtol=rtol,atol=atol)
    return {'schema_coverage':len(columns),'within_numeric_tolerance':int(good.sum()),'total':len(columns),
            'rtol':rtol,'atol':atol,'max_absolute_error':float(np.max(np.abs(actual-expected))),
            'numeric_pair_pass':bool(good.all()),'audio_inference_enabled':False,
            'scope':'One supplied pair only; does not verify recording identity, rights, cohort parity or independent audio accuracy.'}


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser(__doc__)
    for n in ['reference','extracted','columns','out']:p.add_argument('--'+n,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    result=compare(read(a.reference),read(a.extracted),read(a.columns))
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2)
