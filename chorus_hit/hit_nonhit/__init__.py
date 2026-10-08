"""Separate, evidence-gated charted versus non-charted research task."""
TASK = 'hit_nonhit_v1'
LABEL_VERSION = 'us_hot100_release_to_cutoff_v1'
GROUP_VERSION = 'connected_artist_work_album_match_audio_v1'
EXTRACTOR = 'shared-librosa-518-v2'
LABELS = {0: 'Verified non-charted through cutoff', 1: 'Charted on US weekly Hot 100'}
