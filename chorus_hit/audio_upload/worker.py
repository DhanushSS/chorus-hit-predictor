"""Private bounded decoder: input stays in an isolated temporary directory."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def analyze(path):
    with sf.SoundFile(path) as stream:
        if stream.format != 'MP3': raise ValueError('The content is not an MP3; changing a file extension does not convert audio.')
        if not 8000 <= stream.samplerate <= 48000 or stream.channels not in (1, 2):
            raise ValueError('Use mono/stereo MP3 at 8–48 kHz.')
        expected = stream.frames
        if expected / stream.samplerate > 480: raise ValueError('Audio duration limit is 480 seconds.')
        if expected * stream.channels * 4 > 192 * 1024 * 1024: raise ValueError('Decoded audio exceeds the 192 MB budget.')
        total = 0; energy = 0.; peak = 0.; envelope = []
        for block in stream.blocks(blocksize=16384, dtype='float32', always_2d=True):
            total += len(block)
            if total > 480 * stream.samplerate or total * stream.channels * 4 > 192 * 1024 * 1024:
                raise ValueError('Decoded duration/size limit exceeded.')
            if not np.isfinite(block).all(): raise ValueError('Decoded audio contains invalid values.')
            energy += float(np.sum(block.astype('float64') ** 2) / stream.channels)
            peak = max(peak, float(np.max(np.abs(block))))
            envelope.append(float(np.max(np.abs(block))))
        if total == 0: raise ValueError('MP3 has no decodable audio frames.')
        if total + int(.2 * stream.samplerate) < expected: raise ValueError('MP3 appears truncated or incomplete.')
        rms = float(np.sqrt(energy / total))
        if rms < 1e-6: raise ValueError('Audio is silent or too quiet for meaningful analysis.')
        return {'status':'PREVIEW_ONLY', 'duration_seconds':total/stream.samplerate,
                'sample_rate':stream.samplerate, 'channels':stream.channels,
                'decoded_bytes':total*stream.channels*4, 'rms':rms, 'peak':peak,
                'waveform_envelope':envelope, 'prediction':None,
                'message':'Audio prediction is not validated/available. No hit score was calculated.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('input'); parser.add_argument('output'); args = parser.parse_args()
    try: result = analyze(args.input)
    except sf.LibsndfileError:
        result = {'error': 'MP3 could not be decoded. The file is corrupt, truncated or unsupported.'}
    except (ValueError, OSError, RuntimeError) as error:
        result = {'error': ' '.join(str(error).split())[:300]}
    Path(args.output).write_text(json.dumps(result, allow_nan=False))
