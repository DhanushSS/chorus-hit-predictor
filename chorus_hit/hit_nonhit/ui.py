"""Optional research view; default legacy deployment remains unchanged."""
from pathlib import Path
import tempfile
from .artifacts import load, predict_features
from .audio import contract, extract
from .common import read
from chorus_hit.config import ROOT


def available_runs():
    values = []
    for m in sorted((ROOT / 'results/hit_nonhit/v1').glob('*/manifest.json')):
        try:
            load(m.parent, require_evaluated=True)
            values.append(m.parent)
        except (ValueError, OSError, KeyError, TypeError):
            continue
    return values


def render(st):
    with st.expander('Hit vs Non-Hit (experimental)'):
        st.write('Academic target: year-end Hot 100 hits versus songs from comparable artists/albums that are absent from the audited public Hot 100 archive through 30 December 2023.')
        progress = ROOT / 'data/hit_nonhit_v1/academic_progress.json'
        if progress.exists():
            try:
                status = read(progress)
                st.write(f"Existing positive candidates verified: {status['verified_legacy_candidates']} of 366. Complete weekly issues checked: {status['complete_chart_weeks']}.")
                st.caption('Public metadata supports a bounded historical study. It is not an authenticated Billboard export; uncertain identities are excluded.')
                intake = ROOT / 'data/hit_nonhit_v1/audio_intake.csv'
                if intake.exists():
                    st.download_button('Download the audio preparation list', intake.read_bytes(), 'audio_intake.csv', 'text/csv')
            except (OSError, ValueError, KeyError, TypeError):
                st.info('Academic metadata progress is unavailable. Restore the checked-in metadata files to see the preparation list.')
        runs = available_runs()
        if not runs:
            st.info('Dataset/model not ready. Add permitted local recordings for both classes and review their recording identity and chorus positions. Both classes will be extracted with the same settings. No new-target accuracy is available.')
            st.caption('The existing model below uses year-end hits versus other charted songs. Its scores do not validate this new task.')
            return
        run = st.selectbox('Verified research run', runs, format_func=lambda p: p.name)
        b, m = load(run, require_evaluated=True)
        st.caption(f"Observation cutoff: {m['chart_cutoff']} · cohort: {m.get('cohort', 'any weekly hit versus non-charted')} · shared 15-second audio extractor · locked historical evaluation")
        st.warning('Retrospective classification. Scores are uncalibrated and do not predict future commercial success. An automatic repeated excerpt is not a human-verified chorus.')
        uploaded = st.file_uploader('Authorized audio for the experimental task', type=['wav', 'flac', 'ogg', 'mp3'], key='new_task_audio')
        if st.button('Classify with the experimental model', disabled=uploaded is None):
            try:
                with tempfile.TemporaryDirectory(prefix='hit_nonhit_') as temp:
                    path = Path(temp) / ('audio' + Path(uploaded.name).suffix.lower())
                    path.write_bytes(uploaded.getvalue()); X, _, meta = extract(path)
                    result = predict_features(b, X, contract())
                st.metric('Retrospective classification', result['labels'][0]); st.caption(f"Score: {result['scores'][0]:.3f} · {result['score_semantics']['kind']}")
                st.json({'excerpt_method': meta['selection_method'], 'start_seconds': meta['segment_start_seconds'], 'chart_cutoff': result['chart_cutoff']})
            except (ValueError, OSError, RuntimeError) as error:
                st.error(f'Could not classify this recording: {error}')
