"""
Build a single PDF presentation from figures in the `figures/` folder.
Each image will be placed on its own page with a caption.

Run: python -m src.build_presentation
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "data" / "figures_presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "wearable_analysis_summary.pdf"

# Ordered list of figure files and captions
items = [
    (FIG_DIR / "phase1_valid_day_histograms.png", "Phase 1: Valid-day coverage histograms by wearable signal."),
    (FIG_DIR / "coverage_by_variable.png", "Top 20 variables by median valid days (coverage diagnostic)."),
    (FIG_DIR / "phase2_r2_comparison.png", "Phase 2: Baseline vs. augmented R² across behavioral_summary outcomes."),
    (FIG_DIR / "exploratory_r2_lift_histogram.png", "Exploratory: Distribution of R² lifts (augmented - baseline) for fine-grained outcomes."),
    (FIG_DIR / "top_spearman_correlations.png", "Post-hoc: Top 10 absolute Spearman correlations between means and behavior outcomes."),
]

with PdfPages(OUT_PDF) as pdf:
    for path, caption in items:
        if not path.exists():
            print(f"Skipping missing figure: {path}")
            continue
        img = plt.imread(path)
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis("off")
        # caption box
        fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("Wrote presentation:", OUT_PDF)
