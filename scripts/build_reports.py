"""Generate the one-page and two-page write-ups from measured results."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "results/metrics.json").read_text())
PROJECT = json.loads((ROOT / "project.json").read_text())
WIN = next(r for r in M["models"] if r["selected"])
BASE = next(r for r in M["models"] if r["model"] == "Majority baseline")
TEAL = colors.HexColor("#127C80")
INK = colors.HexColor("#172D37")
PALE = colors.HexColor("#EFF5F5")
S = getSampleStyleSheet()
S.add(ParagraphStyle(name="ProjectTitle", fontName="Helvetica-Bold", fontSize=20, leading=23, textColor=INK, spaceAfter=8))
S.add(ParagraphStyle(name="ProjectSub", fontName="Helvetica", fontSize=9.2, leading=13, textColor=INK, spaceAfter=9))
S.add(ParagraphStyle(name="H", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=TEAL, spaceBefore=11, spaceAfter=5))
S.add(ParagraphStyle(name="B", fontName="Helvetica", fontSize=10, leading=13.5, textColor=INK, spaceAfter=6))
S.add(ParagraphStyle(name="SmallB", fontName="Helvetica", fontSize=8.2, leading=10.8, textColor=INK, spaceAfter=4))


def p(text, style="B"):
    return Paragraph(text, S[style])


def header(one_page=False):
    names = "<br/>".join(f"{a['name']} - {a['usn']}" for a in PROJECT["authors"])
    return [p(PROJECT["title"], "ProjectTitle"),
            p(f"{names}<br/>{PROJECT['course']} | September 2026", "ProjectSub")]


def table(rows, widths):
    t = Table([[p(str(v), "SmallB") for v in row] for row in rows], colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), PALE), ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LINEBELOW", (0,0), (-1,0), .8, TEAL),
        ("LINEBELOW", (0,1), (-1,-1), .3, colors.HexColor("#DCE5E6")),
        ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(TEAL); canvas.setLineWidth(.6)
    canvas.line(42, 37, A4[0]-42, 37)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(INK)
    canvas.drawString(42, 24, "Chorus-based classification | Reproducible experiment")
    canvas.drawRightString(A4[0]-42, 24, str(doc.page))
    canvas.restoreState()


def build(path, story):
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=42, rightMargin=42, topMargin=36,
                      bottomMargin=48, title=PROJECT["title"],
                      author="; ".join(a["name"] for a in PROJECT["authors"])).build(
                          story, onFirstPage=footer, onLaterPages=footer)


def main():
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    split = M["split"]
    interval = M["selected_test_ci95_artist_bootstrap"]["balanced_accuracy"]
    story = header()
    story += [p("Problem statement and scope", "H"),
              p("Can 15-second chorus audio features distinguish songs with greater chart success? "
                "This project implements the chorus-based classification idea in Eric Liu's 2021 CS229 report [1]. "
                "It compares six model families and tests whether learned patterns transfer to unfamiliar artists."),
              p("The available public dataset uses a <b>year-end-hit proxy</b>: label 1 contains source year-end Hot 100 selections, "
                "while label 0 contains other songs sampled from weekly Hot 100 charts. Both classes can have charted. "
                "This differs from the reference paper's charted versus uncharted definition and is not an exact replication."),
              p("Dataset and audio features", "H"),
              p("We reuse Antonios Malak's Apache-2.0 feature dataset [2], pinned to commit 838e76f. "
                "It contains <b>751 songs, 71 artist names, and 518 numerical features</b>: 366 label-1 and 385 label-0 songs. "
                "The upstream collection spans 2006-2021. The local audit found no missing feature values, constant columns, "
                "or exact duplicate feature vectors. We remove audio paths and exclude all metadata from model inputs."),
              table([["Feature family", "Channels", "Summary values"],
                     ["Chroma STFT, CQT, CENS", "12 each", "252 total"],
                     ["MFCC (timbre)", "20", "140"],
                     ["Spectral contrast; tonnetz", "7; 6", "49; 42"],
                     ["RMS, centroid, bandwidth, rolloff, zero-crossing rate", "1 each", "35 total"],
                     ["Total", "74", "518"]], [330,70,111]),
              p("Each channel contributes skewness, minimum, maximum, standard deviation, mean, median, and kurtosis. "
                "The source states that pychorus selected 15-second excerpts. Original recordings are not bundled, so "
                "the extraction and label assignments cannot be fully re-audited here.", "SmallB"),
              p("Approach and implementation", "H"),
              p(f"A fixed artist-grouped split allocates <b>{split['train_songs']} songs from {split['train_artists']} artists</b> "
                f"to training and <b>{split['test_songs']} songs from {split['test_artists']} different artists</b> to testing. "
                "Five artist-disjoint folds inside training select hyperparameters and the model by mean balanced accuracy. "
                "The seed is 42. No test scores guide selection."),
              p("A scikit-learn Pipeline fits median imputation, variance filtering, and standardization within each training fold. "
                "PCA retains 95% of training variance for logistic regression, LDA, three SVM kernels, and a neural network. "
                "Random forest and gradient boosting retain the original features. A majority-class classifier provides a baseline. "
                "The saved model remains fitted on training data only."),
              p("Python modules cover data checks, training, evaluation, inference, and audio processing. A Streamlit interface "
                "supports held-out song prediction, results inspection, and experimental uploaded-audio inference. "
                "A notebook and README explain the workflow."),
              PageBreak(),
              p("Measured results and interpretation", "ProjectTitle"),
              p("Balanced accuracy averages recall across both labels. F1 refers to the year-end-hit class. "
                "Every test score below uses the same unseen-artist holdout."),
              table([["Model", "CV BA", "Test BA", "Accuracy", "F1", "AUC"]] +
                    [[r["model"] + (" *" if r["selected"] else ""),
                      f"{r['cv_balanced_accuracy']:.3f}", f"{r['test_balanced_accuracy']:.3f}",
                      f"{r['test_accuracy']:.3f}", f"{r['test_f1']:.3f}", f"{r['test_roc_auc']:.3f}"]
                     for r in M["models"]], [166,69,69,69,69,69]),
              p("* Selected using training cross-validation. Other test results are descriptive comparisons, not a reason to switch models after viewing the holdout.", "SmallB"),
              p("Conclusion", "H"),
              p(f"The selected {M['selected_model']} achieved <b>{WIN['test_balanced_accuracy']:.1%} test balanced accuracy</b> "
                f"and {WIN['test_f1']:.3f} F1, compared with {BASE['test_balanced_accuracy']:.1%} balanced accuracy for the baseline. "
                f"The approximate 95% interval is {interval[0]:.1%}-{interval[1]:.1%}, obtained by resampling whole test artists "
                "2,000 times. It includes 50%. <b>This experiment provides no reliable evidence that these chorus features "
                "predict year-end-hit status for unfamiliar artists.</b> The result does not establish that chorus quality causes popularity."),
              p("Limitations and improvements", "H"),
              p("The dataset is small and historically sampled, and its labels are a proxy inherited from upstream collection code. "
                "Artist grouping uses provided names and cannot fully resolve collaborations. Original audio is unavailable, "
                "so feature-to-recording correctness remains unverified. The upload demo uses a new repetition heuristic or manual "
                "selection with current librosa, rather than exactly reproducing the old extraction environment."),
              p("A stronger follow-up would independently verify chart labels, collect licensed recordings for charted and "
                "uncharted tracks under a fixed time window, re-extract every clip consistently, and evaluate on later release dates. "
                "Further model tuning should use a new evaluation split or nested cross-validation."),
              p("Demonstration and reproducibility", "H"),
              p("Run the local app, choose a held-out song, compare prediction with its dataset label, then inspect the "
                "results table and error analysis. Uploaded audio can demonstrate segment selection and feature extraction, "
                "but its predictions are exploratory. The repository includes a pinned data source, dependency versions, "
                "split manifest, all CV trials, test predictions, tests, and the training-only model."),
              p("References", "H"),
              p('[1] Eric Liu. <i>Predicting Hit Songs Using Repeated Chorus</i>, CS229, 2021. '
                '<link href="https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf" color="#127C80">Stanford report</link>.<br/>'
                '[2] Antonios Malak. <i>Predicting-Hit-Songs-Using-Repeated-Chorus</i>, commit 838e76f. '
                '<link href="https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus" color="#127C80">Dataset and collection notebooks</link>. Apache-2.0.<br/>'
                '[3] Scikit-learn documentation, <i>Common pitfalls and recommended practices</i>. '
                '<link href="https://scikit-learn.org/stable/common_pitfalls.html" color="#127C80">Data leakage and pipelines</link>.', "SmallB")]
    build(out / "Project_Report_2_Pages.pdf", story)
    short = header(True) + [
        p("Problem and dataset", "H"),
        p("We test whether 15-second chorus features can distinguish stronger chart success, following Eric Liu's CS229 project. "
          "The experiment uses a public Apache-2.0 dataset of <b>751 songs from 71 artists</b>, with 518 numerical audio features. "
          "There are 366 year-end-hit labels and 385 other-chart-song labels. This public dataset defines a hit proxy "
          "different from the paper's charted/uncharted target. Its original audio is unavailable."),
        p("Method and implementation", "H"),
        p("Eleven audio feature families produce 74 channels, each summarized by seven statistics. A fixed split holds out "
          "154 songs from 19 artists, leaving 597 songs from 52 artists for training. Five artist-disjoint training folds "
          "select model settings by balanced accuracy. Preprocessing fits inside each fold. PCA keeps 95% of training variance "
          "for non-tree models. We compare logistic regression, LDA, three SVM kernels, random forest, gradient boosting, "
          "a neural network, and a majority-class baseline."),
        p("Measured results", "H"),
        table([["Metric", "Selected polynomial SVM", "Majority baseline"],
               ["Training CV balanced accuracy", f"{WIN['cv_balanced_accuracy']:.1%}", "50.0%"],
               ["Test balanced accuracy", f"{WIN['test_balanced_accuracy']:.1%}", "50.0%"],
               ["Test accuracy", f"{WIN['test_accuracy']:.1%}", f"{BASE['test_accuracy']:.1%}"],
               ["Test F1 / ROC-AUC", f"{WIN['test_f1']:.3f} / {WIN['test_roc_auc']:.3f}", "0.000 / 0.500"]], [191,175,145]),
        p(f"The 95% interval for selected-model balanced accuracy is {interval[0]:.1%}-{interval[1]:.1%}, "
          "using 2,000 bootstrap resamples of test artists. The selected model stays unchanged after test evaluation.", "SmallB"),
        p("Conclusion and demonstration", "H"),
        p("The experiment finds <b>no reliable predictive advantage over the baseline for unfamiliar artists</b>. "
          "This is an empirical limitation of the current data and features, not proof about all music. Historical sampling, "
          "inherited proxy labels, and unavailable original recordings limit interpretation. Future work should verify labels "
          "and consistently re-extract licensed audio before further modeling."),
        p("The working Streamlit demo predicts held-out songs, displays model comparisons and errors, and supports exploratory "
          "audio uploads with automatic or manual 15-second selection. The repository includes the code, notebook, dependency "
          "versions, trained model, data provenance, split manifest, and reproducible results."),
        p("Sources", "H"),
        p('<link href="https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf" color="#127C80">Eric Liu, CS229 report (2021)</link>. '
          '<link href="https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus" color="#127C80">Antonios Malak, dataset and notebooks</link> '
          '(Apache-2.0, commit 838e76f). Full provenance and method details appear in the repository.', "SmallB")]
    build(out / "Project_Summary_1_Page.pdf", short)
    from pypdf import PdfReader
    for name, expected in [("Project_Report_2_Pages.pdf", 2), ("Project_Summary_1_Page.pdf", 1)]:
        actual = len(PdfReader(out / name).pages)
        assert actual == expected, f"{name}: expected {expected} pages, got {actual}"
        print(name, actual, "pages")


if __name__ == "__main__":
    main()
