"""Current Chorus Hit Predictor demo: the evaluated HSP-S feature model."""
from pathlib import Path
import pandas as pd
import streamlit as st
from chorus_hit.config import ROOT
from chorus_hit.hit_nonhit.common import read
from chorus_hit.hsp_s.evaluate import load_model, predict_frame

st.set_page_config(page_title='Chorus Hit Predictor', page_icon='🎵', layout='wide')
st.title('Chorus Hit Predictor')
st.caption('Dhanush Sai Suprapadha · PES2UG24CS154  |  Deepthi V · PES2UG24CS150')
st.write('Explore Billboard chart association using the evaluated HSP-S audio-feature model.')
st.info('This study uses whole-recording Essentia features. Upload matching feature CSVs; MP3 and 15-second chorus prediction are not supported by this model.')
run = ROOT / 'results/hsp_s/run_001'
try:
    bundle = load_model(run)
    result = read(run / 'locked_evaluation.json')
except (ValueError, OSError, KeyError, TypeError) as error:
    st.warning('The verified model is not available yet. Run `python scripts/setup_model.py` from the repository folder with the pinned Python 3.12 environment.')
    st.stop()
metrics = result['metrics']
for col, label, value in zip(st.columns(4), ['Test accuracy', 'Balanced accuracy', 'Hit precision', 'Test songs'],
                             [f"{metrics['accuracy']:.2%}", f"{metrics['balanced_accuracy']:.2%}", f"{metrics['precision']:.2%}", str(result['rows'])]):
    col.metric(label, value)
st.caption('RBF SVM · 436 scalar audio features · 527 held-out identity groups · test used once after development selection')
with st.expander('What the results mean'):
    st.write('Labels describe chart entries in the HSP-S publisher benchmark. Missing chart entries do not prove a song never charted. These retrospective scores are not future-hit probabilities.')
    st.write('95% group-bootstrap interval for balanced accuracy: 70.20–76.08%. Dataset and recording-source bias remain limitations.')
    st.json({'held_out_metrics': metrics, 'group_interval': result['group_interval']})
st.subheader('Classify feature records')
st.write('Upload 1–100 rows in the exact column order from the template. Use the same whole-recording Essentia feature schema as HSP-S.')
st.download_button('Download feature template', ','.join(bundle['columns']) + '\n', 'hsp_s_feature_template.csv', 'text/csv')
upload = st.file_uploader('Audio-feature CSV (maximum 5 MB)', type=['csv'])
if st.button('Classify records', disabled=upload is None):
    try:
        if upload.size > 5 * 1024 * 1024:
            raise ValueError('Use a feature CSV smaller than 5 MB')
        prediction = predict_frame(run, pd.read_csv(upload))
        table = pd.DataFrame({'Classification': prediction['labels'], 'Uncalibrated score': prediction['scores']})
        st.dataframe(table, hide_index=True)
        st.download_button('Download predictions', table.to_csv(index=False), 'predictions.csv', 'text/csv')
    except (ValueError, OSError, KeyError, TypeError) as error:
        st.error(f'Could not classify the file: {error}')
st.divider()
st.caption('Source: Vötter, Mayerl, Specht and Zangerle (2021) · HSP-S · DOI 10.5281/zenodo.5383858 · CC BY 4.0')
st.markdown('[Results and methods](https://github.com/DhanushSS/chorus-hit-predictor/blob/main/docs/hit_nonhit/hsp_s_results/README.md) · [Archived chorus project](https://github.com/DhanushSS/chorus-hit-predictor-archive-2026-10-11)')
