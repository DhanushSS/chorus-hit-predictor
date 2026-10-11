"""MP3 intake uses a killable decoder and always removes temporary uploads."""
from pathlib import Path
import os
import json
import subprocess
import sys
import tempfile
from chorus_hit.config import ROOT
from chorus_hit.supervision import stop_process_group

MAX_AUDIO_BYTES = 25 * 1024 * 1024


def preview_mp3(payload, filename, *, timeout=25, temporary_parent=None):
    if os.name != 'posix': raise ValueError('Bounded MP3 decoding currently supports macOS and Linux only. CSV prediction remains available.')
    if Path(filename).suffix.lower() != '.mp3': raise ValueError('Choose an MP3 file. WAV/FLAC support is not enabled.')
    if not payload: raise ValueError('The MP3 file is empty.')
    if len(payload) > MAX_AUDIO_BYTES: raise ValueError('MP3 upload limit is 25 MB.')
    with tempfile.TemporaryDirectory(prefix='hit-audio-', dir=temporary_parent) as directory:
        folder = Path(directory); source = folder/'input.mp3'; output = folder/'analysis.json'
        source.write_bytes(payload)
        child = subprocess.Popen([sys.executable, '-m', 'chorus_hit.audio_upload.worker', str(source), str(output)],
                                 cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            try: code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error: raise ValueError('Audio decoding timed out. Try a shorter, valid MP3.') from error
        finally: stop_process_group(child)
        if code or not output.exists(): raise ValueError('Audio decoder failed; the file may be corrupt or unsupported.')
        result = json.loads(output.read_text())
        if 'error' in result:
            # Do not reveal temporary paths from native decoder exceptions.
            raise ValueError(result['error'].replace(str(folder), '[temporary audio]'))
        return result
