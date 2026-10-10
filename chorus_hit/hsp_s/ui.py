"""Separate feature-table demo. Never accepts audio under an incompatible extractor."""
from pathlib import Path
import pandas as pd
from chorus_hit.config import ROOT
from chorus_hit.hit_nonhit.common import read
from .evaluate import load_model, predict_frame


def render(st):
    st.title('Hit vs Non-Hit feature study')
    st.write('A separate study using HSP-S, a published dataset of audio descriptors and Billboard chart information.')
    st.info('This model uses whole-recording Essentia features. It does not analyze uploaded MP3s or repeated choruses, and its score is not a forecast of future success.')
    st.caption('Dataset: Vötter, Mayerl, Specht and Zangerle (2021), DOI 10.5281/zenodo.5383858 · CC BY 4.0. Class 0 means no chart entry in the publisher benchmark, not verified never-charted status.')
    runs = []
    for p in sorted((ROOT / 'results/hsp_s').glob('*/manifest.json')):
        try:
            load_model(p.parent); runs.append(p.parent)
        except (ValueError, OSError, KeyError, TypeError): continue
    if not runs:
        st.info('A compatible, evaluated HSP-S model is not available yet. The original Chorus Hit Predictor remains on the main page.')
        return
    run = st.selectbox('Evaluated experiment', runs, format_func=lambda p: p.name)
    bundle = load_model(run); summary = read(run / 'summary.json'); result = read(run / 'locked_evaluation.json')
    columns = st.columns(3)
    for c, title, value in zip(columns, ['Held-out accuracy', 'Balanced accuracy', 'Held-out songs'],
                              [f"{result['metrics']['accuracy']:.1%}", f"{result['metrics']['balanced_accuracy']:.1%}", str(result['rows'])]): c.metric(title, value)
    st.caption(f"Model: {summary['selected_candidate']['id']} · {result['groups']} held-out connected groups · fixed group split")
    st.write('The held-out test was used once after model selection. Viewing its saved results does not retrain or retune the model.')
    with st.expander('Evaluation details'):
        st.json({'nested_development': summary['nested_development'], 'locked_test': result['metrics'], 'group_interval': result['group_interval']})
    st.subheader('Classify matching feature records')
    st.write('Upload 1–100 rows of pre-extracted HSP-S scalar audio features, in the exact column order shown by the template. Artist names, popularity counts and chart ranks are not prediction inputs.')
    st.download_button('Download the feature-column template', ','.join(bundle['columns']) + '\n', 'hsp_s_feature_template.csv', 'text/csv')
    upload = st.file_uploader('Matching audio-feature CSV', type=['csv'])
    if st.button('Classify feature records', disabled=upload is None):
        try:
            if upload.size > 5 * 1024 * 1024: raise ValueError('Use a feature CSV smaller than 5 MB')
            frame = pd.read_csv(upload)
            output = predict_frame(run, frame)
            st.dataframe(pd.DataFrame({'Classification': output['labels'], 'Uncalibrated score': output['scores']}), hide_index=True)
        except (ValueError, OSError, KeyError, TypeError) as error: st.error(f'Could not classify these features: {error}')
