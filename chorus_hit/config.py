from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "chorus_features.csv"
RESULTS = ROOT / "results"
MODEL_PATH = ROOT / "results" / "v2" / "v4_std74_001" / "model.joblib"
SEED = 42
SAMPLE_RATE = 22050
CHORUS_SECONDS = 15
MAX_AUDIO_SECONDS = 480
FEATURE_GROUPS = {
    "chroma_stft": 12, "chroma_cqt": 12, "chroma_cens": 12,
    "mfcc": 20, "rms": 1, "spectral_centroid": 1,
    "spectral_bandwidth": 1, "spectral_contrast": 7,
    "spectral_rolloff": 1, "tonnetz": 6, "zero_crossing_rate": 1,
}
# 'kew' is the upstream spelling of skewness. Preserve the schema for inference.
STATISTICS = ("kew", "min", "max", "std", "mean", "median", "kurtosis")
FEATURE_COLUMNS = [
    f"{family}_{stat}_{channel}"
    for family, channels in FEATURE_GROUPS.items()
    for channel in range(channels)
    for stat in STATISTICS
]
LABELS = {0: "Other chart song", 1: "Year-end hit"}
SOURCE_REPO = "https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus"
SOURCE_COMMIT = "838e76f96f7aa755a882b4d2581afe1e0ad3f8a9"
