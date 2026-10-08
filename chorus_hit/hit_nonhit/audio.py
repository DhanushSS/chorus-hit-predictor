"""One pinned extractor for both labels, with explicit excerpt and rights metadata."""
from pathlib import Path
import time
import librosa
import numpy as np
import soundfile as sf
from chorus_hit import audio as shared
from chorus_hit.config import FEATURE_COLUMNS, SAMPLE_RATE
from . import EXTRACTOR
from .common import digest, sha
from .schema import require


def contract():
    return {'extractor_version': EXTRACTOR, 'settings': shared.extraction_config(EXTRACTOR),
            'shared_code_sha256': sha(shared.__file__), 'wrapper_code_sha256': sha(__file__),
            'feature_schema_hash': digest(FEATURE_COLUMNS), 'feature_columns': FEATURE_COLUMNS,
            'decoder': 'soundfile float32; soxr_hq resampling; arithmetic channel mean'}


def extract(path, start=None, annotation=None):
    path = Path(path)
    info = sf.info(path)  # reject unsupported/corrupt codecs; no deprecated decoder fallback
    require(info.format in {'WAV', 'FLAC', 'OGG', 'MP3'}, 'Unsupported decoded audio format')
    allowed = {'.wav': 'WAV', '.flac': 'FLAC', '.ogg': 'OGG', '.mp3': 'MP3'}
    require(allowed.get(path.suffix.lower()) == info.format, 'Audio extension does not match codec')
    require(15 <= info.duration <= 480, 'Recording must be 15 to 480 seconds')
    began = time.perf_counter()
    decoded, original_sr = sf.read(path, dtype='float32', always_2d=True)
    require(np.isfinite(decoded).all(), 'Decoded audio contains NaN or infinity')
    y = np.mean(decoded, axis=1, dtype=np.float32)
    if original_sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=original_sr, target_sr=SAMPLE_RATE, res_type='soxr_hq')
    sr = SAMPLE_RATE
    segment = shared.select_segment(y, sr, start)
    require(sr == SAMPLE_RATE and segment.audio.ndim == 1 and len(segment.audio) == 15 * SAMPLE_RATE, 'Mono/sample/segment contract mismatch')
    features = shared.extract_features(segment.audio, sr, extractor_version=EXTRACTOR)
    reviewed = bool(annotation and annotation.get('reviewer') and annotation.get('evidence') and annotation.get('is_chorus') is True)
    if reviewed:
        require(start is not None and abs(float(annotation['start_seconds']) - segment.start_seconds) <= .5, 'Human chorus annotation start mismatch')
    method = 'manual_reviewed' if reviewed else 'repetition_heuristic' if segment.repetition_similarity is not None else 'fallback_excerpt'
    # A conservative candidate signature, NOT a guaranteed acoustic duplicate detector.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=2048)
    fingerprint = np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])
    fingerprint /= max(np.linalg.norm(fingerprint), 1e-12)
    meta = {'audio_sha256': sha(path), 'recording_duration': info.duration, 'audio_original_format': info.format,
            'original_sample_rate': info.samplerate, 'original_channels': info.channels,
            'extractor_version': EXTRACTOR, 'extractor_config_hash': digest(contract()),
            'segment_start_seconds': segment.start_seconds, 'segment_start_sample': round(segment.start_seconds * sr),
            'segment_length_seconds': 15, 'segment_samples': len(segment.audio), 'sample_rate': sr,
            'selection_method': method, 'chorus_annotation_status': 'reviewed' if reviewed else 'unverified',
            'selection_confidence_or_similarity': segment.repetition_similarity,
            'perceptual_signature': fingerprint.tolist(), 'signature_policy': 'chroma_moments_cosine_0.995_conservative_v1',
            'feature_row_hash': digest(features.iloc[0].tolist()), 'features_schema_hash': digest(FEATURE_COLUMNS),
            'extraction_seconds': time.perf_counter() - began}
    return features, segment, meta
