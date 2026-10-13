"""Current application; presentation code is separate from frozen model identity."""
import pandas as pd
import streamlit as st
from chorus_hit.config import ROOT
from chorus_hit.hit_nonhit.common import read
from chorus_hit.hsp_s.evaluate import load_model, predict_frame
from chorus_hit.ui_components.csv_input import parse_csv
from chorus_hit.audio_upload.service import preview_mp3

TITLE = 'Billboard Hit Prediction Using Audio Features'
st.set_page_config(page_title=TITLE, page_icon='🎵', layout='wide')
st.title(TITLE)
st.caption('Dhanush Sai Suprapadha · PES2UG24CS154  |  Deepthi V · PES2UG24CS150')
st.write('A reproducible study of chart association from whole-song acoustic descriptors. Originally developed as Chorus Hit Predictor.')
st.info('CSV classification uses the evaluated 436-feature model. MP3 uploads support local audio preview and analysis; MP3 classification is not supported until extractor compatibility is validated.')
run = ROOT / 'results/hsp_s/run_001'
try:
    bundle = load_model(run)
    result = read(run / 'locked_evaluation.json')
except (ValueError, OSError, KeyError, TypeError):
    st.warning('The verified model is not available yet. Run `python scripts/setup_model.py` with the pinned Python 3.12 environment.')
    st.stop()

metrics = result['metrics']
for col, label, value in zip(st.columns(4), ['Historical test accuracy', 'Balanced accuracy', 'Hit precision', 'Held-out songs'],
                             [f"{metrics['accuracy']:.2%}", f"{metrics['balanced_accuracy']:.2%}", f"{metrics['precision']:.2%}", str(result['rows'])]):
    col.metric(label, value)
st.caption('HSP-S · whole-recording Essentia descriptors · RBF SVM · original test used once · no new independent test')
overview, csv_tab, audio_tab, evaluation, about = st.tabs(['Overview', 'Predict · CSV', 'Audio · MP3 preview', 'Model evaluation', 'About'])
with overview:
    st.subheader('What this project answers')
    st.write('Can acoustic measurements distinguish songs with a Billboard entry in the HSP-S benchmark from songs without a publisher-matched entry?')
    st.markdown('**Feature CSV → strict validation → verified SVM → class + decision margin**')
    st.success('CSV: evaluated model and verified sample predictions.')
    st.warning('MP3: preview only. Schema agreement alone cannot prove that extracted features match the training distribution.')
    st.write('Class 0 means no chart entry in this benchmark. It does not establish that a recording never charted, and this app does not forecast future commercial success.')
    st.caption('Demo data — seen during training. The included four-row sample tests functionality, not independent accuracy.')
with csv_tab:
    st.subheader('Classify matching feature records')
    st.write('Upload a UTF-8 CSV: 436 numeric feature columns, 1–100 rows, up to 5 MB. Missing cells use the original trained imputer; an entirely empty row is rejected.')
    left, right = st.columns(2)
    left.download_button('Download feature template', ','.join(bundle['columns'])+'\n', 'hsp_s_template.csv', 'text/csv')
    sample = (ROOT/'data/example_features.csv').read_bytes()
    right.download_button('Download demonstration CSV', sample, 'example_features.csv', 'text/csv')
    st.caption('Demo data — seen during training. Do not interpret sample predictions as a new accuracy result.')
    upload = st.file_uploader('Matching audio-feature CSV', type=['csv'], key='csv_upload', max_upload_size=5)
    reorder = st.checkbox('Explicitly reorder columns when all 436 unique names match', value=False)
    if st.button('Validate and classify CSV', disabled=upload is None, type='primary'):
        try:
            frame = parse_csv(upload.getvalue(), bundle['columns'], reorder=reorder)
            st.success(f'Validated {len(frame)} rows × {len(frame.columns)} features.')
            with st.expander('Input preview'): st.dataframe(frame.head(5), hide_index=True)
            prediction = predict_frame(run, frame)
            output = pd.DataFrame({'Record':range(1,len(frame)+1),'Classification':prediction['labels'],'Decision margin':prediction['scores']})
            st.dataframe(output, hide_index=True)
            st.bar_chart(output.set_index('Record')['Decision margin'], x_label='Record', y_label='Decision margin (not probability)')
            st.download_button('Download predictions', output.to_csv(index=False), 'predictions.csv', 'text/csv')
        except (ValueError, OSError, KeyError, TypeError) as error:
            st.error(str(error))
    st.caption('Positive margin → chart-entry class; negative margin → no-entry class. Distance from zero is an uncalibrated model score, not confidence, probability or causal songwriting advice.')
with audio_tab:
    st.subheader('Local MP3 analysis')
    st.warning('Audio prediction is not validated/available. This mode never generates a hit label or score.')
    st.write('MP3 only · up to 25 MB · 480 seconds · mono/stereo · 8–48 kHz. Audio stays local and temporary; no music is sent to an external API.')
    permitted = st.checkbox('I am allowed to analyze this recording locally.', value=False)
    audio = st.file_uploader('MP3 recording', type=['mp3'], key='mp3_upload', max_upload_size=25)
    if st.button('Analyze MP3', disabled=audio is None or not permitted):
        try:
            with st.spinner('Checking format and decoding within resource limits…'):
                analysis = preview_mp3(audio.getvalue(), audio.name)
            st.audio(audio.getvalue(), format='audio/mpeg')
            a,b,c = st.columns(3)
            a.metric('Duration', f"{analysis['duration_seconds']:.1f} s")
            b.metric('Sample rate', f"{analysis['sample_rate']:,} Hz")
            c.metric('Channels', analysis['channels'])
            st.line_chart(pd.DataFrame({'Peak amplitude envelope':analysis['waveform_envelope']}), y_label='Amplitude', x_label='Audio block')
            st.caption(f"RMS amplitude: {analysis['rms']:.4f}. Descriptive signal analysis only.")
            st.info(analysis['message'])
        except (ValueError, OSError, RuntimeError) as error:
            st.error(str(error))
with evaluation:
    st.subheader('Historical evaluation · test consumed once')
    st.write('73.07% accuracy and 73.17% balanced accuracy were measured on 1,493 publisher feature records, never on uploaded MP3s. New experiments are development-only.')
    st.caption('95% connected-group bootstrap interval for balanced accuracy: 70.20–76.08%.')
    l,r = st.columns(2)
    l.image(str(ROOT/'docs/hit_nonhit/hsp_s_results/confusion_matrix.png'), caption='553 TN · 226 FP · 176 FN · 538 TP')
    r.image(str(ROOT/'docs/hit_nonhit/hsp_s_results/model_comparison.png'), caption='Original development-only comparison of 27 configurations')
    st.write('False positives: 226 no-entry records classified as chart entries. False negatives: 176 chart-entry records missed. Source/codec bias and incomplete aliases remain limitations.')
    with st.expander('Saved metrics and uncertainty'): st.json({'metrics':metrics,'group_interval':result['group_interval']})
with about:
    st.subheader('Methodology and team')
    st.write('7,728 accepted songs · 3,865 chart entries / 3,863 no-entry records · 436 scalar audio features. Related artist, recording and album identities stay together across folds.')
    st.write('Dhanush Sai Suprapadha — PES2UG24CS154\n\nDeepthi V — PES2UG24CS150')
    st.markdown('[HSP-S source · CC BY 4.0](https://zenodo.org/records/5383858) · [Project repository](https://github.com/DhanushSS/chorus-hit-predictor) · [Archived repeated-chorus study](https://github.com/DhanushSS/chorus-hit-predictor-archive-2026-10-11)')
    st.caption('HSP-S: Vötter, Mayerl, Specht and Zangerle (2021). The whole-recording benchmark is separate from the archived repeated-chorus study and Eric Liu’s experimental protocol.')
