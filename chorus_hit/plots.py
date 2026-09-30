"""Figures generated solely from the recorded experiment."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import RESULTS


def make_plots(summary, frame, winner, X_train, records):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    directory = RESULTS / "figures"
    directory.mkdir(exist_ok=True)
    ordered = sorted(summary["models"], key=lambda r: r["cv_balanced_accuracy"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(ordered))
    ax.barh(y-.17, [r["cv_balanced_accuracy"] for r in ordered], height=.32, label="Training CV", color="#127C80")
    ax.barh(y+.17, [r["test_balanced_accuracy"] for r in ordered], height=.32, label="Unseen-artist test", color="#E39738")
    ax.axvline(.5, color="#71808F", linestyle="--", linewidth=1)
    ax.set(yticks=y, yticklabels=[r["model"] for r in ordered], xlim=(0, 1), xlabel="Balanced accuracy",
           title="Model comparison (selection uses training CV)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(directory / "model_comparison.png", dpi=180); plt.close(fig)
    row = next(r for r in summary["models"] if r["selected"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    ConfusionMatrixDisplay(np.array(row["confusion_matrix"]), display_labels=["Other", "Year-end hit"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(summary["selected_model"] + ": test errors")
    RocCurveDisplay.from_predictions(records.label, records.score, ax=axes[1], name="Selected model",
                                     curve_kwargs={"color": "#127C80"})
    axes[1].plot([0, 1], [0, 1], "--", color="#71808F")
    axes[1].set_title("Ranking performance on held-out songs")
    fig.tight_layout(); fig.savefig(directory / "test_evaluation.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    counts = frame.label.value_counts().sort_index()
    bars = ax.bar(["Other chart song", "Year-end hit"], counts, color=["#E39738", "#127C80"], width=.55)
    ax.bar_label(bars, padding=4); ax.set(ylabel="Songs", ylim=(0, max(counts)*1.2), title="Dataset label balance")
    fig.tight_layout(); fig.savefig(directory / "class_balance.png", dpi=180); plt.close(fig)
    if "pca" in winner.named_steps:
        pca = winner.named_steps["pca"]
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), color="#127C80")
        ax.axhline(.95, linestyle="--", color="#E39738")
        ax.set(xlabel="Principal components", ylabel="Cumulative explained variance",
               title="PCA fitted on training songs only", ylim=(0, 1.02))
        fig.tight_layout(); fig.savefig(directory / "pca_variance.png", dpi=180); plt.close(fig)
