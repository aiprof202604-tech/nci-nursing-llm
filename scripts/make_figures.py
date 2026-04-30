"""
Generate publication-quality figures (Figures 1-4) for the NCI manuscript.

Design principles (Tufte / Nature / Economist style):
  - Maximum data-ink ratio: minimal chartjunk, no boxes around plots.
  - Direct labelling on lines; legend used only when unavoidable.
  - Title + subtitle structure that tells the finding, not just describes axes.
  - Annotations highlighting key data points and the missing Claude T=1.5 cell.
  - Larger typography (14-18pt) for readability when embedded in body text.
  - Colorblind-safe palette tuned for print: navy / sienna / teal.

Outputs (in figures/):
  Figure1_NCI_by_Temperature.{png,pdf}
  Figure2_NCI_by_Category.{png,pdf}
  Figure3_NCI_distribution.{png,pdf}
  Figure4_NCI_vs_Accuracy.{png,pdf}
"""

import math
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ─── Typography & global style ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 19,
    "figure.dpi": 110,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "lines.solid_capstyle": "round",
})

COLORS = {
    "gpt":    "#1f3a68",
    "claude": "#b85432",
    "gemini": "#2a8a78",
}
SOFT = {
    "gpt":    "#c4cfdc",
    "claude": "#ecc8b8",
    "gemini": "#bcdbd4",
}
LABEL = {
    "gpt": "GPT-4o",
    "claude": "Claude Opus 4.5",
    "gemini": "Gemini 2.5 Flash-Lite",
}
INK   = "#222222"
MUTED = "#777777"


def calc_nci(answers):
    n = len(answers)
    if n == 0:
        return float("nan")
    counts = Counter(answers)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
    return 1.0 - h / math.log2(4)


def load_data():
    df = pd.read_csv(DATA / "raw_responses.csv")
    df_valid = df[df["is_valid"]].copy()
    rows = []
    for (qid, cat, model, temp), g in df_valid.groupby(
        ["question_id", "category", "model", "temperature"]
    ):
        rows.append({
            "question_id": qid, "category": cat,
            "model": model, "temperature": temp,
            "NCI": calc_nci(g["response"].tolist()),
            "accuracy": (g["response"] == g["intended_answer"].iloc[0]).mean(),
            "n_valid": len(g),
        })
    return df_valid, pd.DataFrame(rows)


def add_title_block(fig, title, subtitle, x=0.04, y_title=0.965, y_sub=0.925):
    fig.text(x, y_title, title, fontsize=19, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(x, y_sub, subtitle, fontsize=13, color=MUTED,
             ha="left", va="top")


def add_source_line(fig, text, x=0.04, y=0.025):
    fig.text(x, y, text, fontsize=10, color=MUTED, ha="left", va="bottom",
             style="italic")


def style_axis(ax, x_grid=False, y_grid=True):
    if y_grid:
        ax.yaxis.grid(True, color="#ececec", linewidth=0.8, zorder=0)
    if x_grid:
        ax.xaxis.grid(True, color="#ececec", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=4, width=0.8)


# ─── Figure 1 ──────────────────────────────────────────────────────────────
def figure1(nci_df):
    fig = plt.figure(figsize=(11, 6.8))
    ax = fig.add_axes([0.10, 0.16, 0.70, 0.66])

    temps_full = [0.0, 0.5, 1.0, 1.5]
    end_y = {}
    for model in ["gpt", "claude", "gemini"]:
        sub = nci_df[nci_df["model"] == model]
        agg = sub.groupby("temperature")["NCI"].agg(["mean", "sem"]).reset_index()
        x = agg["temperature"].values
        y = agg["mean"].values
        e = agg["sem"].values
        c = COLORS[model]
        ax.fill_between(x, y - e, y + e, color=SOFT[model], alpha=0.5,
                        zorder=2, linewidth=0)
        ax.plot(x, y, "-", color=c, linewidth=2.6, zorder=4)
        ax.plot(x, y, "o", color=c, markersize=9, markeredgecolor="white",
                markeredgewidth=1.6, zorder=5)
        end_y[model] = (x[-1], y[-1])

    # Direct labels (offset to avoid overlap)
    label_pos = {
        "gpt":    (1.55, end_y["gpt"][1]),
        "gemini": (1.55, end_y["gemini"][1] - 0.0015),
        "claude": (1.13, end_y["claude"][1] + 0.004),
    }
    for model, (lx, ly) in label_pos.items():
        ax.text(lx, ly, LABEL[model], fontsize=13.5, color=COLORS[model],
                fontweight="bold", va="center")

    # Mark missing Claude T=1.5
    ax.plot(1.5, 0.984, marker="x", color=COLORS["claude"], markersize=13,
            markeredgewidth=2.8, zorder=6)
    ax.annotate("API limit:\nClaude rejects T > 1.0",
                xy=(1.5, 0.984), xytext=(1.30, 0.945),
                fontsize=11, color=COLORS["claude"], style="italic",
                ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=COLORS["claude"],
                                lw=0.9, connectionstyle="arc3,rad=0.2"))

    ax.axhline(1.0, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(1.92, 1.001, "NCI = 1 (perfect)", fontsize=10.5, color=MUTED,
            ha="right", va="bottom", style="italic")

    ax.set_xticks(temps_full)
    ax.set_xticklabels([f"T = {t}" for t in temps_full])
    ax.set_xlim(-0.10, 1.95)
    ax.set_ylim(0.918, 1.012)
    ax.set_yticks([0.92, 0.94, 0.96, 0.98, 1.00])
    ax.set_xlabel("Sampling temperature", labelpad=8)
    ax.set_ylabel("Mean Nursing Consistency Index (NCI)", labelpad=8)
    style_axis(ax)

    add_title_block(
        fig,
        title="Output consistency degrades with temperature, sharply for GPT-4o",
        subtitle="Mean NCI across 30 nursing scenarios per cell. Shaded bands = ± 1 SE.",
    )
    add_source_line(
        fig,
        "Friedman tests: GPT-4o χ²(3) = 13.36, p = .0039    "
        "Gemini χ²(3) = 13.50, p = .0037    "
        "Claude χ²(2) = 3.71, p = .156",
    )

    fig.savefig(FIG / "Figure1_NCI_by_Temperature.png")
    fig.savefig(FIG / "Figure1_NCI_by_Temperature.pdf")
    plt.close(fig)
    print("Saved: Figure1_NCI_by_Temperature.{png,pdf}")


# ─── Figure 2 ──────────────────────────────────────────────────────────────
def figure2(nci_df):
    """Grouped bar chart: three categories on x-axis, three model bars per category."""
    fig = plt.figure(figsize=(11, 6.4))
    ax = fig.add_axes([0.10, 0.18, 0.78, 0.62])

    cats = ["Knowledge", "Priority", "Ethical"]
    pivot = (
        nci_df.groupby(["model", "category"])["NCI"]
        .mean()
        .unstack()
        .reindex(index=["gpt", "claude", "gemini"], columns=cats)
    )
    sd_pivot = (
        nci_df.groupby(["model", "category"])["NCI"]
        .std()
        .unstack()
        .reindex(index=["gpt", "claude", "gemini"], columns=cats)
    )

    bar_w = 0.26
    x_pos = np.arange(len(cats))
    offsets = {"gpt": -bar_w, "claude": 0.0, "gemini": +bar_w}

    for model in ["gpt", "claude", "gemini"]:
        means = pivot.loc[model].values
        sds = sd_pivot.loc[model].values
        c = COLORS[model]
        bars = ax.bar(x_pos + offsets[model], means, bar_w,
                      color=c, edgecolor="white", linewidth=1.4,
                      label=LABEL[model], zorder=3)
        # Value label above each bar
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    fontsize=11, color=c, fontweight="bold")

    # Highlight GPT-4o Ethical bar
    gpt_eth = pivot.loc["gpt", "Ethical"]
    ax.annotate(
        "GPT-4o drops sharply\non ethical scenarios",
        xy=(2 + offsets["gpt"], gpt_eth + 0.005),
        xytext=(1.30, 0.92),
        fontsize=12, color=COLORS["gpt"], style="italic", ha="center",
        fontweight="regular",
        arrowprops=dict(arrowstyle="->", color=COLORS["gpt"], lw=1.1,
                        connectionstyle="arc3,rad=-0.20"))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cats, fontsize=14)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0.86, 1.02)
    ax.set_yticks([0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    ax.set_ylabel("Mean NCI (pooled across temperatures)", labelpad=8)
    ax.set_xlabel("Judgement category", labelpad=8)
    style_axis(ax)
    ax.axhline(1.0, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=1)

    # Custom legend at top, horizontal
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005),
                    ncol=3, frameon=False, fontsize=12.5,
                    handlelength=1.3, columnspacing=2.0, handletextpad=0.6)

    add_title_block(
        fig,
        title="Ethical scenarios drive GPT-4o's inconsistency",
        subtitle="Mean NCI averaged over 10 scenarios × 4 temperatures per cell "
                 "(3 temperatures for Claude).",
        y_title=0.97, y_sub=0.93,
    )
    add_source_line(
        fig,
        "Kruskal–Wallis on scenario-level means: H(2) = 3.61, p = .16 (omnibus n.s.). "
        "Pattern is descriptive and model-specific.",
    )

    fig.savefig(FIG / "Figure2_NCI_by_Category.png")
    fig.savefig(FIG / "Figure2_NCI_by_Category.pdf")
    plt.close(fig)
    print("Saved: Figure2_NCI_by_Category.{png,pdf}")


# ─── Figure 3 ──────────────────────────────────────────────────────────────
def figure3(nci_df):
    fig = plt.figure(figsize=(13, 7.2))
    ax = fig.add_axes([0.06, 0.20, 0.74, 0.58])

    tiers = [
        ("Perfect (NCI = 1.0)",         lambda v: v >= 0.99999, "#2a8a78"),
        ("High (0.8 ≤ NCI < 1.0)",      lambda v: 0.8 <= v < 0.99999, "#a4d4c8"),
        ("Moderate (0.5 ≤ NCI < 0.8)",  lambda v: 0.5 <= v < 0.8, "#e9c46a"),
        ("Low (NCI < 0.5)",             lambda v: v < 0.5, "#c25b35"),
    ]

    cells = [
        ("gpt", 0.0), ("gpt", 0.5), ("gpt", 1.0), ("gpt", 1.5), None,
        ("claude", 0.0), ("claude", 0.5), ("claude", 1.0), ("claude", 1.5), None,
        ("gemini", 0.0), ("gemini", 0.5), ("gemini", 1.0), ("gemini", 1.5),
    ]

    bar_w = 0.72
    x_positions = []
    bar_labels = []

    for i, cell in enumerate(cells):
        if cell is None:
            continue
        model, temp = cell
        sub = nci_df[(nci_df["model"] == model) & (nci_df["temperature"] == temp)]
        x_positions.append(i)
        bar_labels.append(f"T = {temp}")

        if len(sub) == 0:
            # No bar drawn — empty column with explicit "no data" marker.
            # A 100%-height hatched bar visually suggests a quantity; a marker does not.
            ax.plot([i - bar_w / 2, i + bar_w / 2], [0, 0],
                    color="#999999", linewidth=2.5, solid_capstyle="butt",
                    clip_on=False)
            ax.text(i, 8, "no data\n(API limit:\nT > 1.0\nnot supported)",
                    ha="center", va="bottom",
                    fontsize=10, color="#888888", style="italic",
                    linespacing=1.3)
            continue

        bottom = 0
        for label, predicate, color in tiers:
            n = sum(predicate(v) for v in sub["NCI"])
            pct = n / len(sub) * 100
            if pct > 0:
                ax.bar(i, pct, width=bar_w, bottom=bottom, color=color,
                       edgecolor="white", linewidth=1.4)
                if pct >= 9:
                    txt_color = "white" if color in ("#2a8a78", "#c25b35") else "#222222"
                    ax.text(i, bottom + pct / 2, f"{pct:.0f}%",
                            ha="center", va="center",
                            fontsize=11, color=txt_color, fontweight="bold")
            bottom += pct

    group_centers = {"GPT-4o": 1.5, "Claude Opus 4.5": 7.5,
                     "Gemini 2.5 Flash-Lite": 12.5}
    group_ranges = {"GPT-4o": (0, 3), "Claude Opus 4.5": (5, 8),
                    "Gemini 2.5 Flash-Lite": (10, 13)}
    name_to_color = {
        "GPT-4o": COLORS["gpt"],
        "Claude Opus 4.5": COLORS["claude"],
        "Gemini 2.5 Flash-Lite": COLORS["gemini"],
    }
    for name, xc in group_centers.items():
        x0, x1 = group_ranges[name]
        ax.plot([x0 - 0.4, x1 + 0.4], [-7, -7], color=INK, lw=1.2,
                clip_on=False)
        ax.text(xc, -13, name, ha="center", va="top", fontsize=14,
                fontweight="bold", color=name_to_color[name])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, fontsize=11.5)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.7, 13.7)
    ax.set_ylabel("Percentage of scenarios", labelpad=8)
    ax.set_xlabel("")
    style_axis(ax)
    ax.tick_params(axis="x", length=0)

    handles = []
    for label, _, color in tiers:
        handles.append(mpatches.Patch(facecolor=color, edgecolor="white",
                                      linewidth=1.2, label=label))
    # Custom legend handle for the "no data" stub
    no_data_handle = Line2D([0], [0], color="#999999", linewidth=2.5,
                            solid_capstyle="butt", label="No data (API limit)")
    handles.append(no_data_handle)
    leg = ax.legend(handles=handles, loc="center left",
                    bbox_to_anchor=(1.01, 0.5), frameon=False,
                    fontsize=11.5, title="NCI tier", title_fontsize=12.5,
                    handlelength=1.6, handleheight=1.1)
    leg.get_title().set_fontweight("bold")

    add_title_block(
        fig,
        title="Most scenarios are perfectly consistent — the imperfect tail grows with temperature",
        subtitle="Per-scenario NCI distribution within each cell (30 scenarios per cell).",
    )
    add_source_line(
        fig,
        "Tiers chosen for narrative clarity. The 'Low' tier (NCI < 0.5) is "
        "near-uniform — the model effectively guesses among the four options.",
    )

    fig.savefig(FIG / "Figure3_NCI_distribution.png")
    fig.savefig(FIG / "Figure3_NCI_distribution.pdf")
    plt.close(fig)
    print("Saved: Figure3_NCI_distribution.{png,pdf}")


# ─── Figure 4 ──────────────────────────────────────────────────────────────
def figure4(nci_df):
    fig = plt.figure(figsize=(14, 5.8))

    models = ["gpt", "claude", "gemini"]
    correlations = {
        "gpt":    (0.503, "p < .001"),
        "claude": (0.991, "p < .001"),
        "gemini": (0.074, "p = .42"),
    }

    panel_w = 0.255
    panel_h = 0.56
    starts = [0.06, 0.385, 0.71]

    for i, model in enumerate(models):
        ax = fig.add_axes([starts[i], 0.18, panel_w, panel_h])
        sub = nci_df[nci_df["model"] == model]
        c = COLORS[model]

        ax.plot([0, 1], [0, 1], color="#cccccc", linewidth=1.0,
                linestyle="--", zorder=1)

        # Main scatter
        ax.scatter(sub["NCI"], sub["accuracy"], color=c, alpha=0.55, s=60,
                   edgecolor="white", linewidth=0.6, zorder=3)

        # Reliably wrong outliers
        wrong = sub[(sub["NCI"] >= 0.99999) & (sub["accuracy"] == 0)]
        if len(wrong) > 0:
            ax.scatter(wrong["NCI"], wrong["accuracy"], facecolor="white",
                       edgecolor=c, s=140, linewidth=2.2, zorder=4)
            n = len(wrong)
            ax.annotate(
                f"Reliably wrong\n(n = {n} cell{'s' if n > 1 else ''})",
                xy=(1.0, 0.0), xytext=(0.55, 0.18),
                fontsize=10.5, color=c, style="italic", ha="center",
                arrowprops=dict(arrowstyle="->", color=c, lw=0.9,
                                connectionstyle="arc3,rad=-0.2"))

        r, pstr = correlations[model]
        # Two-line title: model name on top, r value below
        ax.text(0.0, 1.13, LABEL[model],
                transform=ax.transAxes,
                fontsize=15, fontweight="bold", color=c,
                ha="left", va="bottom")
        ax.text(0.0, 1.045, f"Pearson r = {r:.2f}    {pstr}",
                transform=ax.transAxes,
                fontsize=12, color=MUTED, ha="left", va="bottom")

        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("NCI", labelpad=6)
        if i == 0:
            ax.set_ylabel("Accuracy", labelpad=8)
        style_axis(ax, x_grid=True, y_grid=True)
        ax.set_aspect("equal", adjustable="box")

    add_title_block(
        fig,
        title="Consistency and accuracy are not the same — the relationship is model-specific",
        subtitle="Each dot = one (scenario × temperature) cell. Hollow circles = NCI = 1.0 with accuracy = 0 ('reliably wrong').",
        x=0.06, y_title=0.96, y_sub=0.91,
    )
    add_source_line(
        fig,
        "Dashed line: NCI = accuracy reference. "
        "GPT-4o n = 120 cells; Claude n = 90 (T = 1.5 unavailable); Gemini n = 120.",
        x=0.06,
    )

    fig.savefig(FIG / "Figure4_NCI_vs_Accuracy.png")
    fig.savefig(FIG / "Figure4_NCI_vs_Accuracy.pdf")
    plt.close(fig)
    print("Saved: Figure4_NCI_vs_Accuracy.{png,pdf}")


def cleanup_old():
    old = [
        "Figure2_NCI_heatmap.png", "Figure2_NCI_heatmap.pdf",
        "Figure3_NCI_decomposition.png", "Figure3_NCI_decomposition.pdf",
    ]
    for f in old:
        p = FIG / f
        if p.exists():
            p.unlink()
            print(f"Removed superseded: {f}")


def main():
    np.random.seed(42)
    _, nci_df = load_data()
    cleanup_old()
    figure1(nci_df)
    figure2(nci_df)
    figure3(nci_df)
    figure4(nci_df)


if __name__ == "__main__":
    main()
