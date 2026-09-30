"""Run with: streamlit run app.py"""
import io
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from chorus_hit.config import FEATURE_COLUMNS, LABELS, MODEL_PATH, RESULTS, ROOT
from chorus_hit.data import load_data
from chorus_hit.train import model_scores

st.set_page_config(page_title="Chorus Lab | Hit Song Prediction", page_icon="🎵", layout="wide")
st.markdown("""<style>
.block-container {max-width:1160px; padding-top:2.4rem; padding-bottom:3rem;}
h1 {letter-spacing:-1.4px;} [data-testid="stMetricValue"] {font-size:1.85rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def assets():
    bundle = joblib.load(MODEL_PATH)
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    data = load_data()
    predictions = pd.read_csv(RESULTS / "test_predictions.csv")
    return bundle, metrics, data, predictions


st.caption("UE24CS352A · MACHINE LEARNING MINI-PROJECT")
st.title("Can a chorus predict a hit?")
st.write("Explore what a model can learn from 15 seconds of music.")
try:
    bundle, metrics, data, predictions = assets()
except FileNotFoundError:
    st.error("Train the project first: python -m chorus_hit.train")
    st.stop()

winner = next(r for r in metrics["models"] if r["selected"])
st.info("Here, a ‘hit’ means a year-end Billboard hit in the source dataset. The other class contains other charting songs. This demo does not forecast future commercial success.")
columns = st.columns(4)
for column, label, value in zip(columns, ["Songs", "Chorus features", "Selected model", "Test balanced accuracy"],
                              [str(len(data)), "518", "Poly SVM" if bundle["model_name"] == "Polynomial SVM" else bundle["model_name"], f"{winner['test_balanced_accuracy']:.1%}"]):
    column.metric(label, value)

demo, results_tab, method = st.tabs(["Try the model", "Results & evidence", "How the project works"])

with demo:
    mode = st.radio("Choose a demo", ["Held-out song", "Upload audio"], horizontal=True)
    if mode == "Held-out song":
        st.subheader("Test a song the model did not train on")
        st.write("These songs and their artists were held out during training. Predictions use only their 518 audio features.")
        options = predictions.sort_values(["artist", "title"]).copy()
        lookup = {r.track_id: f"{r.artist} — {r.title}" for r in options.itertuples()}
        track_id = st.selectbox("Song", options.track_id.tolist(), format_func=lambda x: lookup[x])
        if st.button("Predict this song", type="primary"):
            example = data.loc[data.track_id == track_id]
            predicted = int(bundle["pipeline"].predict(example[FEATURE_COLUMNS])[0])
            score, score_type = model_scores(bundle["pipeline"], example[FEATURE_COLUMNS])
            actual = int(example.label.iloc[0])
            left, right = st.columns(2)
            left.metric("Model prediction", LABELS[predicted])
            right.metric("Dataset label", LABELS[actual])
            if predicted == actual:
                st.success("The prediction matches the dataset label for this song.")
            else:
                st.warning("The model misclassified this song. Errors are part of the measured test results.")
            st.caption(f"Model score: {score[0]:.3f} · {score_type}. This is not a calibrated probability of success.")
            with st.expander("Inspect the input features"):
                st.dataframe(example[FEATURE_COLUMNS].T.rename(columns={example.index[0]: "Value"}), use_container_width=True)
            st.caption("This demo uses stored chorus features. The dataset does not include playable source recordings.")
    else:
        st.subheader("Analyze your own audio")
        st.write("Upload a 15-second chorus or a song up to 8 minutes long. WAV, FLAC, and MP3 are supported.")
        st.warning("Experimental: the original recordings and exact extraction environment are unavailable. New-audio predictions have not been validated against those recordings.")
        uploaded = st.file_uploader("Audio file", type=["wav", "flac", "mp3", "ogg"])
        selection = st.radio("Choose the 15-second segment", ["Find a repeated segment", "Set the start time"], horizontal=True)
        start = st.number_input("Start time in seconds", min_value=0.0, value=0.0, step=1.0) if selection == "Set the start time" else None
        if st.button("Analyze audio", type="primary", disabled=uploaded is None):
            from chorus_hit.audio import extract_features, load_audio, select_segment
            suffix = Path(uploaded.name).suffix.lower()
            try:
                with st.spinner("Selecting a segment and measuring its sound…"):
                    with tempfile.TemporaryDirectory(prefix="chorus_lab_") as temp:
                        audio_path = Path(temp) / f"upload{suffix}"
                        audio_path.write_bytes(uploaded.getvalue())
                        y, sr = load_audio(audio_path)
                        segment = select_segment(y, sr, start_seconds=start)
                        X = extract_features(segment.audio, sr)
                        predicted = int(bundle["pipeline"].predict(X)[0])
                        score, score_type = model_scores(bundle["pipeline"], X)
                    audio_bytes = io.BytesIO()
                    sf.write(audio_bytes, segment.audio, sr, format="WAV")
                st.metric("Model prediction", LABELS[predicted])
                st.caption(f"15-second excerpt starts at {segment.start_seconds:.1f}s · {segment.method}")
                st.audio(audio_bytes.getvalue(), format="audio/wav")
                stride = max(1, len(segment.audio)//1000)
                waveform = pd.DataFrame({"Seconds": np.arange(len(segment.audio))[::stride]/sr,
                                         "Amplitude": segment.audio[::stride]}).set_index("Seconds")
                st.line_chart(waveform, height=160, color="#127C80")
                st.caption(f"Model score: {score[0]:.3f} · {score_type}. This score is uncalibrated.")
                if start is None:
                    st.write("Automatic selection finds repeated musical patterns. Listen to the excerpt and use a manual start time if it chose a verse or instrumental section.")
                st.download_button("Download the 518 extracted features", X.to_csv(index=False),
                                   "chorus_features.csv", "text/csv")
            except Exception as error:
                st.error(f"Could not analyze this file: {error}")

with results_tab:
    st.subheader("Evaluation on unfamiliar artists")
    split = metrics["split"]
    st.write(f"Training: {split['train_songs']} songs from {split['train_artists']} artists. "
             f"Testing: {split['test_songs']} songs from {split['test_artists']} different artists. "
             "Model selection uses five folds within the training set, with artists kept separate in every fold.")
    st.caption("Balanced accuracy gives each class equal weight. A majority-class baseline scores 50% on this metric.")
    st.image(str(RESULTS / "figures" / "model_comparison.png"), use_container_width=True)
    comparison = pd.DataFrame(metrics["models"])
    visible = comparison[["model", "cv_balanced_accuracy", "test_accuracy", "test_balanced_accuracy",
                          "test_precision", "test_recall", "test_f1", "test_roc_auc", "selected"]]
    st.dataframe(visible.style.format({c: "{:.3f}" for c in visible.columns if c.startswith(("cv_", "test_"))}),
                 hide_index=True, use_container_width=True)
    st.caption("The selected model was chosen before examining test scores. Test results for all candidates are descriptive comparisons.")
    st.image(str(RESULTS / "figures" / "test_evaluation.png"), use_container_width=True)
    interval = metrics["selected_test_ci95_artist_bootstrap"]["balanced_accuracy"]
    st.write(f"Selected model balanced accuracy: **{winner['test_balanced_accuracy']:.1%}**. "
             f"Approximate 95% interval from resampling test artists: **{interval[0]:.1%}–{interval[1]:.1%}**.")
    with st.expander("All test predictions, including mistakes"):
        st.dataframe(predictions, hide_index=True, use_container_width=True)
        st.download_button("Download test predictions", predictions.to_csv(index=False),
                           "test_predictions.csv", "text/csv")

with method:
    st.subheader("Audio becomes numbers, then a prediction")
    st.markdown("""
1. **Select 15 seconds.** The source dataset used pychorus. The upload demo uses a repetition heuristic or your chosen start time.
2. **Measure the sound.** Eleven feature families describe pitch, timbre, energy, and spectral shape.
3. **Summarize each channel.** Seven statistics turn 74 channels into 518 input values.
4. **Train and compare models.** Imputation, scaling, and optional PCA are fitted inside training folds.
5. **Evaluate once on held-out artists.** A fixed test set measures errors beyond the artists used for training.
""")
    st.write("The model never receives artist names, song titles, file paths, or chart positions as inputs.")
    st.subheader("What this experiment can establish")
    st.write("It measures how well chorus features distinguish the source dataset’s two labels. "
             "It cannot prove that a catchy chorus causes popularity, and it cannot guarantee success for a new song.")
    st.write("Limitations include a small historical sample, source labels that have not been fully re-audited, "
             "unavailable original recordings, and differences between old and current audio extraction software.")
    st.markdown("[Reference paper](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf) · "
                "[Dataset and original extraction notebooks](https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus)")

st.divider()
st.caption("Dhanush Sai Suprapadha · PES2UG24CS154  |  Deepthi V · PES2UG24CS150")
