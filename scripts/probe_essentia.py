"""Generate a local synthetic MP3 and audit optional extractor coverage, not accuracy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chorus_hit.supervision import supervise


def main(output):
    columns=next(csv.reader((ROOT/'data/example_features.csv').open()))
    with tempfile.TemporaryDirectory(prefix='essentia-probe-') as folder:
        audio=Path(folder)/'generated.mp3'; extracted=Path(folder)/'features.json'
        t=np.arange(15*44100)/44100
        wave=(.2*np.sin(2*np.pi*220*t)+.1*np.sin(2*np.pi*440*t))*(.7+.3*np.sin(2*np.pi*2*t))
        sf.write(audio,wave,44100,format='MP3')
        start=time.monotonic()
        try: code=supervise([sys.executable,'-m','chorus_hit.audio_upload.extract_essentia',str(audio),str(extracted)],90,ROOT)
        except subprocess.TimeoutExpired: raise SystemExit('Extractor exceeded the 90-second probe budget')
        if code: raise SystemExit('Optional extractor failed; CSV and MP3 preview are unaffected')
        data=json.loads(extracted.read_text()); features=data.pop('features')
        missing=[key for key in columns if key not in features]
        result={**data,'schema_coverage':len(columns)-len(missing),'required':len(columns),'missing_keys':missing,
                'fixture':'15-second software-generated two-tone amplitude-modulated MP3; not a song or parity reference',
                'fixture_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),'seconds':time.monotonic()-start,
                'platform':sys.platform,'matching_recording_pairs':0,'audio_inference_enabled':False}
        with Path(output).open('x') as handle: json.dump(result,handle,indent=2)
        print(f"Schema coverage: {result['schema_coverage']}/436. Audio inference disabled.")

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',required=True);main(parser.parse_args().out)
