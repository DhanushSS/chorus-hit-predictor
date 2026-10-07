"""15-second segment selection and the source-compatible 518-feature schema.

The automatic selector is a transparent repetition heuristic, not pychorus and
not a verified musical chorus detector. Users can choose the chorus start.
"""
from dataclasses import dataclass
import warnings

import librosa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .config import (CHORUS_SECONDS, FEATURE_COLUMNS, FEATURE_GROUPS, MAX_AUDIO_SECONDS,
                     SAMPLE_RATE, STATISTICS)


@dataclass
class Segment:
    audio: np.ndarray
    start_seconds: float
    method: str
    repetition_similarity: float | None = None


def load_audio(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_AUDIO_SECONDS+1)
    if len(y) > MAX_AUDIO_SECONDS * sr:
        raise ValueError(f"Choose an audio file no longer than {MAX_AUDIO_SECONDS//60} minutes.")
    if len(y) < CHORUS_SECONDS * sr:
        raise ValueError("At least 15 seconds of audio is needed.")
    if not np.isfinite(y).all():
        raise ValueError("The audio contains invalid samples.")
    if np.sqrt(np.mean(y.astype(float)**2)) < 1e-5:
        raise ValueError("The audio is silent or too quiet to analyze.")
    return y, sr


def select_segment(y, sr=SAMPLE_RATE, start_seconds=None):
    y = np.asarray(y)
    if y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(sr) or sr <= 0:
        raise ValueError("Audio must be a finite mono vector at a positive sample rate")
    if np.sqrt(np.mean(y.astype(float)**2)) < 1e-5:
        raise ValueError("Audio is silent or too quiet")
    size = int(CHORUS_SECONDS * sr)
    if len(y) < size:
        raise ValueError("At least 15 seconds of audio is needed.")
    if start_seconds is not None:
        requested_start = float(start_seconds)
        if not np.isfinite(requested_start):
            raise ValueError("Start time must be finite")
        if requested_start < 0:
            raise ValueError("Start time must be nonnegative")
        start = int(round(requested_start*sr))
        if start < 0 or start + size > len(y):
            raise ValueError("The selected 15-second segment extends outside the audio.")
        return Segment(y[start:start+size], start/sr, "Manual 15-second selection")
    if len(y) < 35 * sr:
        start = (len(y)-size)//2
        return Segment(y[start:start+size], start/sr, "Centered excerpt (too short to verify repetition)")
    # Chroma represents pitch classes. Compare the same 15-second pitch sequence
    # at different times, requiring non-overlap. This may also select a verse.
    hop = 2205
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=4096)
    width = int(round(CHORUS_SECONDS * sr/hop))
    starts = np.arange(int(5*sr/hop), chroma.shape[1]-width+1, int(2*sr/hop))
    if len(starts) < 2:
        start = (len(y)-size)//2
        return Segment(y[start:start+size], start/sr, "Centered excerpt (few candidates)")
    windows = np.stack([chroma[:, s:s+width].reshape(-1) for s in starts])
    windows -= windows.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(windows, axis=1, keepdims=True)
    windows = windows / np.maximum(norm, 1e-10)
    similarity = windows @ windows.T
    overlap = np.abs(starts[:, None]-starts[None, :]) < width
    similarity[overlap] = -np.inf
    a, _ = np.unravel_index(np.argmax(similarity), similarity.shape)
    score = float(np.max(similarity))
    if not np.isfinite(score):
        start = (len(y)-size)//2
        return Segment(y[start:start+size], start/sr, "Centered excerpt (no separated repeat)")
    start = min(int(starts[a]*hop), len(y)-size)
    return Segment(y[start:start+size], start/sr, "Automatic repeated-segment candidate", score)


def summarize(matrix):
    values = []
    for row in np.atleast_2d(matrix):
        constant = np.std(row) < 1e-10
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values.extend([0.0 if constant else float(skew(row)), float(np.min(row)),
                           float(np.max(row)), float(np.std(row)), float(np.mean(row)),
                           float(np.median(row)), 0.0 if constant else float(kurtosis(row))])
    return values


def extract_features(segment, sr=SAMPLE_RATE, *, extractor_version="legacy-librosa-518-v1"):
    if extractor_version not in {"legacy-librosa-518-v1", "shared-librosa-518-v2"}:
        raise ValueError("Unknown extractor version")
    y = np.asarray(segment, dtype=np.float32)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    if len(y) != CHORUS_SECONDS * sr:
        raise ValueError("Feature extraction requires an exact 15-second segment.")
    if not np.isfinite(y).all() or np.sqrt(np.mean(y.astype(float)**2)) < 1e-5:
        raise ValueError("Choose a non-silent segment with finite samples.")
    # Explicit legacy padding reduces drift from the upstream librosa 0.8-era code.
    common = {"y": y, "sr": sr, "hop_length": 512}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        arrays = {
            "chroma_stft": librosa.feature.chroma_stft(**common, pad_mode="reflect"),
            "chroma_cqt": librosa.feature.chroma_cqt(**common),
            "chroma_cens": librosa.feature.chroma_cens(**common),
            "mfcc": librosa.feature.mfcc(**common, n_mfcc=20, pad_mode="reflect"),
            "rms": librosa.feature.rms(y=y, frame_length=2048, hop_length=512, pad_mode="reflect"),
            "spectral_centroid": librosa.feature.spectral_centroid(**common, pad_mode="reflect"),
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(**common, pad_mode="reflect"),
            "spectral_contrast": librosa.feature.spectral_contrast(**common, pad_mode="reflect"),
            "spectral_rolloff": librosa.feature.spectral_rolloff(**common, pad_mode="reflect"),
            "tonnetz": librosa.feature.tonnetz(y=y, sr=sr),
            # The upstream positional call zero_crossing_rate(x, sr) used sr as
            # frame_length. Reproduce it deliberately to match the trained data.
            "zero_crossing_rate": librosa.feature.zero_crossing_rate(y, frame_length=sr if extractor_version == "legacy-librosa-518-v1" else 2048, hop_length=512),
        }
    values = []
    for family, width in FEATURE_GROUPS.items():
        if arrays[family].shape[0] != width:
            raise ValueError(f"Unexpected feature dimensions for {family}")
        values.extend(summarize(arrays[family]))
    result = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Audio produced non-finite features. Try a different chorus.")
    return result


def extraction_config(extractor_version="legacy-librosa-518-v1"):
    from importlib.metadata import version
    from .audit import json_hash
    if extractor_version not in {"legacy-librosa-518-v1", "shared-librosa-518-v2"}:
        raise ValueError("Unknown extractor version")
    config={"version":extractor_version,"sample_rate":SAMPLE_RATE,"mono":True,"duration_seconds":CHORUS_SECONDS,
            "normalization":"none beyond decoder mono/resampling","dtype":"float32","hop_length":512,
            "fft_length":2048,"center":True,"stft_padding":"reflect","mfcc_count":20,
            "rms_frame_length":2048,"zcr_frame_length":SAMPLE_RATE if extractor_version.startswith("legacy") else 2048,
            "chroma_channels":12,"contrast_bands":6,"rolloff_percent":.85,"cqt_bins_per_octave":12,
            "skew_bias":True,"kurtosis_fisher":True,"constant_channel_higher_moments":0,
            "remaining_parameters":"Pinned librosa defaults; implementation hash recorded below",
            "repetition_selector":{"version":"chroma-repeat-v1","hop":2205,"fft":4096,"candidate_step_seconds":2,"initial_skip_seconds":5},
            "versions":{p:version(p) for p in ["librosa","numpy","scipy","soundfile"]}}
    config["config_sha256"]=json_hash(config)
    return config


def extract_record(path, start_seconds=None, extractor_version="legacy-librosa-518-v1"):
    from .audit import sha256_file
    from pathlib import Path
    config=extraction_config(extractor_version)
    y,sr=load_audio(path); segment=select_segment(y,sr,start_seconds)
    features=extract_features(segment.audio,sr,extractor_version=extractor_version)
    metadata={"audio_sha256":sha256_file(path),"extractor_config":config,
              "implementation_sha256":sha256_file(Path(__file__)),"segment_start_seconds":segment.start_seconds,
              "segment_duration_seconds":len(segment.audio)/sr,"selection":segment.method,"repetition_similarity":segment.repetition_similarity,"sample_rate":sr,
              "schema_validated":True,"waveform_parity_verified":False}
    return features,segment,metadata
