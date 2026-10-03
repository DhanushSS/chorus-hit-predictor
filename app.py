"""Chorus Lab: validated model demo and transparent development evidence."""
import io
from pathlib import Path
import tempfile
import json
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from chorus_hit.config import ROOT
from chorus_hit.artifacts import active_run,load_run,run_cache_key,RUNS
from chorus_hit.evaluate_run import load_historical

st.set_page_config(page_title='Chorus Lab | Hit Song Prediction',page_icon='🎵',layout='wide')
st.markdown('<style>.block-container{max-width:1160px;padding-top:4rem;padding-bottom:3rem;}h1{letter-spacing:-1.4px;}[data-testid="stMetricValue"]{font-size:1.8rem;}</style>',unsafe_allow_html=True)


@st.cache_resource
def assets(run_id,manifest_hash):
    return load_run(run_id)


st.caption('UE24CS352A · MACHINE LEARNING MINI-PROJECT')
st.title('Can a chorus predict a hit?')
st.write('Explore 15 seconds of music and inspect what the experiments actually found.')
try:
    default=active_run();available=[p.name for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith('.') and (p/'manifest.json').is_file()]
    available=[default]+sorted(x for x in available if x!=default and x!='v2_quick_001')
    with st.sidebar:
        run_id=st.selectbox('Demo run',available,format_func=lambda v: 'Original model (active)' if v==default else {'v2_nested_001':'V2 research candidate','v3_nested_001':'V3 robustness study','v4_std74_001':'V4 chorus-variation study'}.get(v,v))
        st.caption('The original model remains active. Selecting a research candidate here does not promote it.')
    key=run_cache_key(run_id);a=assets(*key);s=a.summary;bundle=a.bundle
except Exception as error:
    st.error(f'Project files are incompatible or incomplete: {error}');st.stop()
features=bundle['features'];labels=bundle['labels'];predictions=a.predictions
config_path=a.path/'config.json'
selected_config=json.loads(config_path.read_text()) if config_path.exists() else None
st.info('Class 1 means a source year-end Billboard hit. Class 0 contains other charting songs. This retrospective experiment does not establish future-hit prediction.')
status=s['evaluation_status'].replace('_',' ')
status_short={'historical_test':'Historical','nested_development':'Nested CV','tuning_development':'Tuning CV'}.get(s['evaluation_status'],status.title())
cols=st.columns(4)
for c,label,value in zip(cols,['Dataset songs','Raw audio features','Evaluation','Balanced accuracy'],
                        [str(len(a.data)),str(len(features)),status_short,f"{s['metrics']['balanced_accuracy']:.1%}"]):c.metric(label,value)
st.caption(f"Run: {run_id} · Model: {bundle['model_name']} · Evidence: {s['evidence_status'].replace('_',' ')}")
if selected_config and selected_config.get('study_note'):
    st.caption('Exploratory follow-up: these development folds were already inspected. The result requires confirmation on new data.')

demo,results_tab,method=st.tabs(['Try the model','Results & evidence','How the project works'])
with demo:
    mode=st.radio('Choose a demo',['Held-out song','Upload audio'],horizontal=True)
    if mode=='Held-out song':
        st.subheader('Try a historical test song')
        st.write(f'These songs and their artist-name groups were excluded from model training. Predictions use {len(features)} audio features. This benchmark has already been inspected.')
        # Membership is checked against the validated run's training IDs and frozen baseline.
        base=assets(*run_cache_key('v1_baseline'));ids=base.predictions.track_id
        options=a.data.loc[a.data.track_id.isin(ids)].sort_values(['artist','title'])
        if set(ids)&set(bundle['train_track_ids']):st.error('Evaluation membership overlaps training');st.stop()
        lookup={r.track_id:f'{r.artist} — {r.title}' for r in options.itertuples()}
        track_id=st.selectbox('Song',options.track_id.tolist(),format_func=lookup.get)
        if st.button('Predict this song',type='primary'):
            row=a.data.loc[a.data.track_id==track_id];prediction,score,semantics=a.predict(row[features]);p=int(prediction[0]);actual=int(row.label.iloc[0])
            left,right=st.columns(2);left.metric('Model prediction',labels[str(p)]);right.metric('Dataset label',labels[str(actual)])
            if p==actual:st.success('The prediction matches the dataset label for this song.')
            else:st.warning('The model misclassified this song. Every error remains in the recorded evaluation.')
            st.caption(f"Score: {score[0]:.3f} · {semantics['kind'].replace('_',' ')} · native threshold {semantics['threshold']}. This is not a calibrated chance of commercial success.")
            with st.expander('Inspect the input features'):st.dataframe(row[features].T.rename(columns={row.index[0]:'Value'}),use_container_width=True)
            st.caption('Stored chorus features are available. Playable source recordings are absent.')
    else:
        st.subheader('Analyze your own audio')
        st.warning('Experimental: source recordings and their exact extraction environment are unavailable. New-audio predictions have no measured end-to-end validation.')
        uploaded=st.file_uploader('Audio file',type=['wav','flac','mp3','ogg'])
        selection=st.radio('Choose the 15-second segment',['Find a repeated segment','Set the start time'],horizontal=True)
        start=st.number_input('Start time in seconds',min_value=0.,value=0.,step=1.) if selection=='Set the start time' else None
        if st.button('Analyze audio',type='primary',disabled=uploaded is None):
            from chorus_hit.audio import extract_record
            try:
                with st.spinner('Measuring the selected excerpt…'),tempfile.TemporaryDirectory(prefix='chorus_') as tmp:
                    path=Path(tmp)/('upload'+Path(uploaded.name).suffix.lower());path.write_bytes(uploaded.getvalue())
                    X,segment,meta=extract_record(path,start,bundle['extractor_version']);prediction,score,semantics=a.predict(X,bundle['extractor_version'])
                st.metric('Model prediction',labels[str(int(prediction[0]))]);audio=io.BytesIO();sf.write(audio,segment.audio,meta['sample_rate'],format='WAV');st.audio(audio.getvalue())
                st.caption(f"Excerpt starts at {segment.start_seconds:.1f}s · {segment.method} · {bundle['extractor_version']}")
                st.caption(f"Score: {score[0]:.3f}. An uncalibrated model score, not a success probability.")
                st.download_button(f'Download the {len(features)} extracted features',X.to_csv(index=False),'chorus_features.csv','text/csv')
                with st.expander('Extraction details'):st.json(meta)
            except Exception as error:st.error(f'Could not analyze this file: {error}')

with results_tab:
    st.subheader('What the evaluation supports')
    ci=s['ci95']['balanced_accuracy']
    st.write(f"**{status.title()}** balanced accuracy: **{s['metrics']['balanced_accuracy']:.1%}**. Approximate artist-bootstrap interval: **{ci[0]:.1%}–{ci[1]:.1%}**.")
    st.write('Artist-name grouping keeps provided names separate. Aliases, guest performers and recording duplicates still need evidence-backed identity resolution.')
    target=s['target_status'];st.write('**Strict >75% target:** '+('met in this evaluation only' if target['target_met_in_this_evaluation'] else 'not met')+'. Fresh-test confirmation: unavailable.')
    st.dataframe(pd.DataFrame([{'Metric':k.replace('_',' ').title(),'Value':v['value'],'Above 75%':v['passed']} for k,v in target['metrics'].items()]),hide_index=True,use_container_width=True)
    if s['evaluation_status']=='historical_test':
        st.caption('Original selection used training cross-validation before the original test evaluation. The published test is now a historical benchmark.')
        st.image(str(a.path/'figures/model_comparison.png'),use_container_width=True)
    else:
        st.caption('Nested predictions assess the selection procedure across outer folds. The final candidate fits all original development songs. These are different fitted models.')
        st.write(f"Full-development tuning BA: {s['tuning_balanced_accuracy']:.1%}. Tuning scores can be optimistic.")
        st.dataframe(pd.read_csv(a.path/'final_development_ranking.csv').sort_values('mean_balanced_accuracy',ascending=False),hide_index=True,use_container_width=True)
    with st.expander('Historical V2 comparison'):
        research=assets(*run_cache_key('v2_nested_001'));h=load_historical(ROOT/'results/evaluations/v2_nested_001_historical')
        st.write(f"V2 nested development BA: **{research.summary['metrics']['balanced_accuracy']:.1%}**. V2 historical BA: **{h['metrics']['balanced_accuracy']:.1%}**.")
        d=h['paired_vs_v1'];st.write(f"Matched historical change: {100*d['balanced_accuracy_difference']:+.1f} percentage points; approximate paired interval {100*d['ci95'][0]:+.1f} to {100*d['ci95'][1]:+.1f} percentage points.")
        st.caption('The paired interval includes zero. The original model stays active. New permitted recordings, verified identities/labels and a genuinely fresh set are still needed.')
    with st.expander('All recorded predictions'):
        st.dataframe(predictions,hide_index=True,use_container_width=True)
        st.download_button('Download predictions',predictions.to_csv(index=False),run_id+'_predictions.csv','text/csv')
    with st.expander('Model and input contract'):
        st.json({'run_id':run_id,'input_features':len(features),'extractor':bundle['extractor_version'],'score':bundle['score_semantics'],'task':a.manifest['task_version'],'groups':a.manifest['group_version']})
with method:
    st.subheader('Audio becomes measurements, then a prediction')
    st.markdown(f'''1. Choose a 15-second excerpt.
2. Compute the model's ordered set of {len(features)} audio measurements.
3. Fit preprocessing and a classifier using only each training fold.
4. Use separate artist groups for model selection and assessment.
5. Report errors, uncertainty and the evaluation's limits.''')
    research_config=selected_config or json.loads((research.path/'config.json').read_text())
    study_name='This research study' if selected_config else 'The V2 research study'
    st.write(f"{study_name} compares {len(research_config['candidates'])} declared settings, including the original controls. Nested evaluation uses {research_config['outer_folds']} outer and {research_config['inner_folds']} inner grouped folds. Historical labels do not tune the research candidate.")
    st.write('The task is year-end hit versus other chart song. Artist names, song titles, file paths and chart positions never enter the predictor.')
    st.write('We have verified the software, data hashes and recorded metrics. Label/recording identity coverage and audio extraction equivalence remain incomplete.')
    st.markdown('[Reference paper](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf) · [Feature dataset](https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus)')
st.divider();st.caption('Dhanush Sai Suprapadha · PES2UG24CS154 | Deepthi V · PES2UG24CS150')
