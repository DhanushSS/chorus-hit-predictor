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
        st.write('Separate research target: US weekly Hot 100 charted recordings versus comparable recordings verified absent through a fixed historical cutoff.')
        runs = available_runs()
        if not runs:
            st.info('Dataset/model not ready. Complete authorized weekly chart history, reviewed recording identities, and lawful audio for both classes are still needed. No new-target accuracy is available.')
            st.caption('The existing model below uses year-end hits versus other charted songs. Its scores do not validate this new task.')
            return
        run = st.selectbox('Verified research run', runs, format_func=lambda p: p.name)
        b, m = load(run, require_evaluated=True)
        st.caption(f"Observation cutoff: {m['chart_cutoff']} · US weekly Hot 100 · shared 15-second audio extractor · locked historical evaluation")
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
