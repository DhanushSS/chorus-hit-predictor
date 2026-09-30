# Start here

Your topic is **Predicting Hit Songs Using Repeated Chorus**.

Imagine giving the computer many 15-second clips. Each clip becomes 518 measurements of its sound. The computer studies labelled examples and tries to predict the label of a different song.

## Run it

1. Double-click **Run Demo.command** in the main project folder on your Mac.
2. Open **Try the model**, choose **Held-out song**, and press **Predict this song**.
3. Compare the model's answer with the dataset label.
4. Open **Results & evidence** to see the measured performance.

The app is available at http://localhost:8501 while it is running. If the browser tab closes, reopen that address. If the server stops, run the launcher again.

## Understand these five points

1. **This is binary classification:** the model chooses one of two classes.
2. **The data are real song features:** 751 songs, 518 features each. We reused a cited public dataset.
3. **Our labels are year-end hit versus other chart song:** both classes may have appeared on Billboard. This differs from the supplied paper.
4. **Training and test artists are separate:** 597 songs train the model; 154 songs from different artists test it.
5. **The result is close to chance:** selected-model balanced accuracy is 46.6%, versus a 50% baseline. We cannot claim reliable hit prediction.

The uploaded-audio mode also works, but those predictions are exploratory because the original source recordings and extraction environment cannot be verified. Use the held-out song mode for the measured experiment.

## What to submit or present

- Private GitHub repository containing the project and README.
- `Project_Report_2_Pages.pdf`, or `Project_Summary_1_Page.pdf` if faculty confirms one page.
- `Project_Presentation.pptx` during the review.
- The working app for the live demo.

Review dates in the supplied guidelines: **October 5-9, 2026**. Submission deadline: **October 10, 2026, 11:59 PM**.

## What you should do before the review

Read the two-page report, run the demo yourselves, and rehearse the questions in `VIVA_AND_DEMO.md`. Both names and USNs are already included. The repository is private; faculty access will need to be added when you have the required account details.
