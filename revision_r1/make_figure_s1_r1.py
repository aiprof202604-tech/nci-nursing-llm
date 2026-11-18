#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary Figure S1 (revision): same design as submitted; panel titles use the item IDs of the revised text
(114-AM-75, 110-PM-71) and percentages show one decimal where needed (72.5%, 87.5%), matching the Results."""
import os, json
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from PIL import Image

OUT = os.environ.get("NCI_OUT", "outputs")
R = json.load(open(os.path.join(OUT, "revision_results.json")))
ex = R["B_pooled_modal_by_item_model"]
for f in ["LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"]:
    fm.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{f}")
mpl.rcParams.update({"font.family": "Liberation Sans", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#222222", "savefig.facecolor": "white"})
M = ["gpt-4o", "claude-opus-4.5", "gemini-2.5-flash-lite"]; LAB = ["GPT-4o", "Claude", "Gemini"]
COL = {"gpt-4o": "#0072B2", "claude-opus-4.5": "#D55E00", "gemini-2.5-flash-lite": "#009E73"}
OK, BAD = "#2E7D32", "#C62828"
panels = [("114_2025-AM-75", "Item 114-AM-75 (2025, Ethical/Legal)", "B"),
          ("110_2021-PM-71", "Item 110-PM-71 (2021, Priority)", "B")]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.8), sharey=True)
for ax, (uid, title, key) in zip(axes, panels):
    ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    for i, m in enumerate(M):
        d = ex[uid][m]; pct = 100 * d["frac"]; ok = d["modal"] == key
        ax.bar(i, pct, width=0.6, color=COL[m], edgecolor=OK if ok else BAD, linewidth=3.2, zorder=3)
        lab = f"{pct:.0f}%" if abs(pct - round(pct)) < 1e-9 else f"{pct:.1f}%"
        ax.text(i, pct + 2.0, f"{d['modal']}\n{lab}", ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=OK if ok else BAD)
    ax.set_xticks(range(3)); ax.set_xticklabels(LAB); ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(f"{title}   key = {key}", fontsize=13.5)
axes[0].set_ylabel("Modal-answer frequency (%)\npooled across temperatures")
fig.legend(handles=[Patch(facecolor="white", edgecolor=OK, linewidth=3, label="Modal answer = official key"),
                    Patch(facecolor="white", edgecolor=BAD, linewidth=3, label="Stable convergence on non-key answer")],
           loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(bottom=0.2, wspace=0.08)
png = os.path.join(OUT, "FigureS1.png"); fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
im = Image.open(png).convert("RGB"); w, h = im.size; im = im.resize((w // 2, h // 2), Image.LANCZOS)
im.convert("CMYK").save(os.path.join(OUT, "FigureS1_CMYK.tif"), dpi=(300, 300), compression="tiff_lzw")
im.save(os.path.join(OUT, "FigureS1_embed.png"), dpi=(300, 300))
print("FigureS1", im.size, "aspect", round(im.size[0] / im.size[1], 3))
