# Understand and present the updated project

## Explain it in one minute

“Our project asks whether a short chorus helps distinguish year-end Billboard hits from other chart songs. We have 751 real songs represented by 518 sound measurements. Both classes contain charting songs, so we clearly define the target.

We first reproduced our original results and fixed problems in provenance and artifact handling. Then we tested 35 declared settings. We kept artists separate and fitted feature selection inside each training fold. Nested cross-validation assessed the whole selection procedure.

The procedure achieved about 52.9% balanced accuracy. Our final development candidate uses logistic regression with 50 features selected by mutual information. It achieved 55.2% balanced accuracy on the old historical benchmark, but uncertainty is large. None of the five required metrics exceeded 75%, and no genuinely fresh test is available. The project demonstrates a working, reproducible research system; it does not establish reliable hit prediction.”

## What changed and why?

- **Provenance:** an audit must hash the file actually loaded. New imports must identify their real source.
- **Artifact consistency:** the model, dataset, summary, labels and features must belong to the same recorded experiment.
- **Feature representations:** PCA preserves variation, while supervised selection looks for relationships with the label. We compared them inside training folds.
- **Nested evaluation:** inner folds choose a configuration; outer folds test that selection procedure on different artists.
- **Honest target checks:** accuracy, balanced accuracy, precision, recall and F1 each must be strictly greater than 0.75. Rounded display values never decide the check.

## Questions you should be able to answer

**Why not select the model with the highest old test score?** That would use test information to choose the model. We select using development folds and label the old test as historical.

**Why keep the original demo model?** The new development interval includes chance, and the paired historical improvement interval includes zero. The V2 model is available as a research option, with no promotion claim.

**Does 55.2% versus 46.6% mean an 8.6% relative improvement?** The balanced-accuracy difference is about 8.54 percentage points. It is a matched historical comparison, with an approximate interval from -1.59 to +16.86 points.

**What does balanced accuracy mean?** It averages recall for class 0 and class 1. The constant majority classifier scores 50% when both classes exist.

**What is mutual information?** A measure of statistical dependence between a feature and the label. We estimate it from fitting rows only and retain 50 features. The final pipeline still accepts all 518 ordered raw inputs.

**Why is nested ROC-AUC blank?** Different outer folds can choose different models with different score scales. We retain each fold's AUC and do not pool those incompatible scores into one headline AUC.

**Do the audio tests prove uploaded songs are accurate?** No. Synthetic audio checks software behavior only. Original recordings and their exact extraction environment are absent, so audio predictions remain exploratory.

**Why didn't we get 75%?** The measured result is below that target. Training/validation gaps, limited artist coverage, imperfect evidence and representations are possible factors. We have not proved one cause, and no particular model or dataset size guarantees 75%.

## Five-minute demo

1. Start the app. Show the active run and its evaluation label.
2. Predict one historical song. Compare its prediction and dataset label.
3. Show that mistakes remain in the results. Explain the model's uncalibrated score.
4. Select the V2 research candidate in the sidebar. Explain 518 input features versus 50 selected features.
5. Show the five target checks, nested interval and historical comparison.
6. End with the concrete missing inputs: permitted recordings, verified identities/labels and a fresh locked evaluation set.

Both students should run the app and understand the code before presenting. This document suggests practice; it does not claim either student completed particular work independently. Implementation and materials were prepared with Codex assistance.
