# Viva and demo guide

## A 45-second explanation

Our project studies whether the sound of a 15-second chorus can indicate a song's chart success. We use a public dataset with 751 songs and 518 chorus features per song. Our labels mean year-end hit and other chart song. We compare several machine learning models, keeping artists separate between training and testing. The selected polynomial SVM scores 46.6% balanced accuracy on unfamiliar artists, compared with a 50% baseline. So our experiment shows that these chorus features alone do not generalize reliably. We built an app that demonstrates the predictions and exposes the actual errors.

## Rehearse a five-minute demonstration

1. **Explain the task and labels, 40 seconds.** State the difference from the paper's labels clearly.
2. **Show the data, 40 seconds.** Point out 751 songs, 518 audio features, and two classes. Explain that titles and artist names are only for display and splitting.
3. **Show a held-out prediction, 60 seconds.** Select a song and predict. If it is wrong, explain that the interface displays the mistake honestly. Do not imply the audio is being played in this mode; it uses stored features.
4. **Show the results, 60 seconds.** Explain the baseline, model selection using training CV, and the untouched artist holdout. Show the confusion matrix.
5. **Show audio processing, 40 seconds.** If you have a local song file, upload it and manually choose a chorus start. Listen to the 15-second selection. State that new-audio predictions are exploratory.
6. **Conclude, 40 seconds.** Explain the negative result and the next improvement: independently verified labels and consistently processed audio.

## Questions you should be able to answer

**What is supervised learning?** We train on examples whose class labels are known. The model learns a relationship between audio measurements and those labels.

**Why use the chorus?** The reference paper asks whether repeated hooks contain useful information about popularity. Restricting the input to a short segment makes that hypothesis testable.

**What is a feature?** A numerical description of sound. RMS describes signal energy; chroma describes pitch classes; MFCCs summarize timbre; spectral measurements describe how energy is distributed across frequencies.

**Why exactly 518 features?** Eleven feature families produce 74 channels. Seven summary statistics per channel give 74 × 7 = 518.

**What is PCA?** It combines correlated input measurements into fewer axes that preserve much of their variation. We retain 95% of training variance. PCA does not use the class label, and it does not guarantee better prediction.

**Why scale features?** Their numerical ranges differ. Scaling prevents large-magnitude measurements from dominating distance-based and linear methods.

**What is data leakage?** Information from test data influences model training or selection. We avoid it by splitting first and fitting preprocessing inside training folds. Source file paths contain label clues, so they are excluded entirely.

**Why separate artists?** A random song split can put the same artist in both sets. The model might learn artist-specific production patterns. Our split asks whether patterns transfer to different artists. The grouping is based on available names, so it is not perfect for collaborations.

**What is cross-validation?** Training data are divided into five folds. For each candidate setting, the model trains on four folds and validates on the remaining fold. We rotate through the folds and average balanced accuracy. No artist appears in both sides of a fold.

**Why balanced accuracy?** It averages the recall of the two classes, giving both equal weight. A classifier that always predicts the majority class has 50% balanced accuracy in this binary task.

**What are precision, recall, F1, and AUC?** Precision asks how often predicted hits are labelled hits. Recall asks how many labelled hits were detected. F1 combines precision and recall. ROC-AUC measures ranking across thresholds; 0.5 is chance-level ranking.

**Why polynomial SVM?** It had the highest mean balanced accuracy in the training CV search. It did poorly on the holdout. That difference illustrates uncertainty and weak generalization.

**Why not choose linear SVM since it has a better test result?** Choosing after viewing test results would use the holdout for model selection. We kept the CV-selected model and reported all comparisons. A future experiment can compare methods with nested CV or a fresh holdout.

**Does a low score mean the code failed?** No. The code trains, predicts, and evaluates correctly. The result says the tested data and features do not support reliable generalization. Do not claim this disproves every possible relationship between music and popularity.

**Did you reproduce the paper exactly?** No. We implemented its chorus-based idea using a separate public dataset and an artist-disjoint evaluation. The labels, sample size, model implementation, and inference extraction differ. These differences are documented.

**Did you collect all the recordings yourselves?** No. We reused and attributed a public feature dataset. Original recordings are not supplied with it. We provide an audio-processing implementation for new local uploads.

**Are the labels fully verified?** No. They follow the upstream collection code. The source's class-0 songs come from sampled weekly charts, so calling them never-charted would be incorrect.

**Is the output a probability of becoming famous?** No. It is an uncalibrated model score for the dataset's classes. It does not include promotion, social trends, artist reach, or future events.

**How does automatic chorus selection work?** It compares non-overlapping 15-second chroma sequences and finds a repeated pattern. A verse can also repeat, so the app calls it a candidate and lets the user choose the start manually.

**What does the uncertainty interval mean?** We repeatedly resample the test artists and recompute the metric. The resulting interval reflects uncertainty within this small held-out sample. It does not cover all sources of bias or model-selection uncertainty.

**What would you improve?** Verify hit definitions against dated chart records, obtain consistent licensed audio, re-extract all choruses using one configuration, enlarge the dataset, and test on later releases and different artists.

## Suggested division for preparation

This is a proposed division, not a claim that these contributions have already happened.

- Dhanush: learn the data pipeline, training code, repository setup, and run the demo.
- Deepthi: learn feature meanings, evaluation, limitations, and present the results.
- Both: run the notebook, explain the label definition, and rehearse all questions.

Record actual contributions after doing the work. Review the AI-assisted implementation and source attribution before submitting.
