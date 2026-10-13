"""Optional isolated MusicExtractor probe; output is NOT approved for inference."""
import argparse
import json
from pathlib import Path

PINNED_VERSION = '2.1b6.dev1389'
PARAMETERS = dict(analysisSampleRate=44100, lowlevelFrameSize=2048, lowlevelHopSize=1024,
                  lowlevelWindowType='blackmanharris62', lowlevelSilentFrames='noise', lowlevelZeroPadding=0,
                  tonalFrameSize=4096, tonalHopSize=2048, tonalWindowType='blackmanharris62',
                  tonalSilentFrames='noise', tonalZeroPadding=0, rhythmMethod='degara',
                  rhythmMinTempo=40, rhythmMaxTempo=208, loudnessFrameSize=88200, loudnessHopSize=44100,
                  startTime=0, endTime=1000000, requireMbid=False, chromaprintCompute=False)
STATISTICS = ['mean','var','median','min','max','dmean','dmean2','dvar','dvar2']


def extract(path):
    try:
        from importlib.metadata import version
        import essentia.standard as es
        import numpy as np
    except ImportError as error:
        raise ValueError('Optional Essentia is unavailable. CSV and MP3 preview remain usable; see audio compatibility documentation.') from error
    if version('essentia') != PINNED_VERSION: raise ValueError('Use the pinned isolated Essentia profile; other versions are not validated.')
    features, _ = es.MusicExtractor(**PARAMETERS, lowlevelStats=STATISTICS, rhythmStats=STATISTICS, tonalStats=STATISTICS)(str(path))
    scalars = {key:float(features[key]) for key in features.descriptorNames()
               if key.startswith(('lowlevel.','rhythm.','tonal.')) and np.isscalar(features[key])
               and not isinstance(features[key],str) and np.isfinite(features[key])}
    return {'profile':'essentia-musicextractor-research-unvalidated-v1','essentia_package':version('essentia'),
            'statistics':STATISTICS,'parameters':PARAMETERS,'features':scalars,'validated_for_prediction':False,
            'metadata':{key:str(features[key]) for key in features.descriptorNames() if key.startswith('metadata.version.')}}


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('input');p.add_argument('output');a=p.parse_args()
    Path(a.output).write_text(json.dumps(extract(a.input),indent=2,allow_nan=False)+'\n')
