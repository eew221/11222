"""Generate the v23 human-reference audit figure with embedded TrueType fonts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


METHODS = ["proposed_lexicographic", "center_inside", "hungarian_iou", "max_iou", "global_pooling"]
DISPLAY = {
    "proposed_lexicographic": "RC-WSSI",
    "center_inside": "Center-inside",
    "hungarian_iou": "Hungarian-IoU",
    "max_iou": "Max-IoU",
    "global_pooling": "Global pooling",
}
COLORS = {
    "proposed_lexicographic": "#D55E00",
    "center_inside": "#56B4E9",
    "hungarian_iou": "#009E73",
    "max_iou": "#E69F00",
    "global_pooling": "#8C8C8C",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    analysis = root / "experiments" / "rc_wssi_human_reference_method_audit_20260812_v1"
    output = root / "paper_mva" / "figures" / "v23"
    output.mkdir(parents=True, exist_ok=True)
    rows = {row["method"]: row for row in read_csv(analysis / "human_reference_method_summary.csv")}

    # Pin a known TrueType face and embed glyphs as Type 42. This avoids
    # renderer-dependent substitution and malformed character extraction.
    font_path = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(METHODS))
    accuracy = np.array([float(rows[m]["box_exact_accuracy"]) for m in METHODS])
    low = np.array([float(rows[m]["filename_group_bootstrap_low"]) for m in METHODS])
    high = np.array([float(rows[m]["filename_group_bootstrap_high"]) for m in METHODS])
    yerr = np.vstack([accuracy - low, high - accuracy])
    axes[0].bar(x, accuracy, color=[COLORS[m] for m in METHODS], edgecolor="white", linewidth=0.5)
    axes[0].errorbar(x, accuracy, yerr=yerr, fmt="none", ecolor="#333333", capsize=3, linewidth=1)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Exact owner-set accuracy")
    axes[0].set_title("Blinded human ownership audit")
    axes[0].set_xticks(x, [DISPLAY[m] for m in METHODS], rotation=20, ha="right")
    axes[0].yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    measures = ["owner_link_precision", "owner_link_recall", "owner_link_f1"]
    measure_labels = ["Precision", "Recall", "F1"]
    width = 0.15
    for index, method in enumerate(METHODS):
        values = [float(rows[method][measure]) for measure in measures]
        axes[1].bar(
            np.arange(len(measures)) + (index - 2) * width,
            values,
            width=width * 0.92,
            label=DISPLAY[method],
            color=COLORS[method],
        )
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Owner-link score")
    axes[1].set_title("Owner-link precision / recall / F1")
    axes[1].set_xticks(np.arange(len(measures)), measure_labels)
    axes[1].yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    axes[1].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.16), columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout(w_pad=1.6)
    pdf_path = output / "fig7_human_reference_assignment_audit.pdf"
    png_path = output / "fig7_human_reference_assignment_audit.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"font={font_path}")
    print(pdf_path)


if __name__ == "__main__":
    main()
