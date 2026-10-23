#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures 1-3 for the Round-1 revision (ACI-2026-07-RA-0313).
Same visual identity as the submitted figures (Okabe-Ito: GPT-4o #0072B2, Claude #D55E00, Gemini #009E73;
markers o / s / ^; palette validated with the dataviz validator: worst adjacent CVD dE 11.0).
Changes: Fig 1 'normalized' (American English), non-overlapping annotation, bands = +/-1 SE;
Fig 2 x-label = modal-answer frequency; Fig 3 = tie-corrected expected capture curves with joint-bootstrap bands.
Outputs: PNG 600 dpi (preview) and TIFF CMYK 300 dpi LZW (submission). Resume: bootstrap bands cached (npz);
parallel: one process per model; progress printed."""
import os, sys, json, time
from multiprocessing import Pool
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revision_analyses as ra

OUT = os.path.join(os.environ.get("NCI_OUT", "outputs"), "figures"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "cache"); os.makedirs(CACHE, exist_ok=True)
for f in ["LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"]:
    fm.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/{f}")
mpl.rcParams.update({"font.family": "Liberation Sans", "font.size": 11, "axes.labelsize": 11.5,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9,
                     "axes.edgecolor": "#222222", "text.color": "#111111", "axes.labelcolor": "#111111",
                     "xtick.color": "#222222", "ytick.color": "#222222", "savefig.facecolor": "white",
                     "pdf.fonttype": 42})
M = ["gpt-4o", "claude-opus-4.5", "gemini-2.5-flash-lite"]
COL = {"gpt-4o": "#0072B2", "claude-opus-4.5": "#D55E00", "gemini-2.5-flash-lite": "#009E73"}
MK = {"gpt-4o": "o", "claude-opus-4.5": "s", "gemini-2.5-flash-lite": "^"}
LAB = {"gpt-4o": "GPT-4o", "claude-opus-4.5": "Claude Opus 4.5", "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite"}
GRID = "#e6e6e6"; REF = "#8c8c8c"
GRIDX = np.linspace(0, 1, 101)

def save(fig, name):
    png = os.path.join(OUT, f"{name}.png"); fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    w, h = im.size; im = im.resize((w // 2, h // 2), Image.LANCZOS)          # 600 -> 300 dpi
    im.convert("CMYK").save(os.path.join(OUT, f"{name}_CMYK.tif"), dpi=(300, 300), compression="tiff_lzw")
    print(f"  saved {name}: {im.size[0]}x{im.size[1]} px @300 dpi", flush=True)

# ------------------------------------------------------------------ expected capture curve (ties random)
def expected_curve(nci, err):
    nci = np.asarray(nci, float); err = np.asarray(err, int); W = err.sum(); n = len(err)
    if W == 0: return None
    e, starts, ends = ra._order_and_groups(nci, err)
    caught = np.zeros(n); before = 0.0
    for s, t in zip(starts, ends):
        size = t - s; eg = e[s:t].sum(); caught[s:t] = before + eg * np.arange(1, size + 1) / size; before += eg
    frac = np.arange(1, n + 1) / n
    return np.interp(GRIDX, np.r_[0, frac], np.r_[0, caught / W])

def model_curves(model, cells):
    cm = cells[cells.model == model]
    curves = [expected_curve(cm[cm.temp == t].NCI.values, (cm[cm.temp == t].modal_correct.values == 0).astype(int))
              for t in ra.TEMPS[model]]
    return np.mean(np.vstack([c for c in curves if c is not None]), axis=0)

def boot_band(model, B=2000, seed=20261022):
    f = os.path.join(CACHE, f"band_{model}.npz")
    if os.path.exists(f):
        d = np.load(f); return d["lo"], d["hi"]                           # resume
    cells = pd.read_pickle(ra.OUT / "cells.pkl")
    cm = cells[cells.model == model]; items = np.array(sorted(cm["item"].unique()))
    by_t = {t: cm[cm.temp == t].set_index("item").loc[items] for t in ra.TEMPS[model]}
    rng = np.random.default_rng(seed + M.index(model)); reps = []; t0 = time.time()
    for b in range(B):
        idx = rng.integers(0, len(items), len(items)); cs = []
        for t in ra.TEMPS[model]:
            d = by_t[t].iloc[idx]
            c = expected_curve(d.NCI.values, (d.modal_correct.values == 0).astype(int))
            if c is not None: cs.append(c)
        if cs: reps.append(np.mean(np.vstack(cs), axis=0))
        if (b + 1) % 500 == 0: print(f"    [{LAB[model]}] bootstrap {b+1}/{B}  {time.time()-t0:.0f}s", flush=True)
    R = np.vstack(reps); lo, hi = np.percentile(R, 2.5, axis=0), np.percentile(R, 97.5, axis=0)
    np.savez(f, lo=lo, hi=hi, n=len(reps)); return lo, hi

# ------------------------------------------------------------------ figures
def fig1(cells):
    fig, ax = plt.subplots(figsize=(6.9, 5.2))
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.axhline(1.0, color=REF, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for m in M:
        cm = cells[cells.model == m]; T = np.array(ra.TEMPS[m])
        mu = np.array([cm[cm.temp == t].NCI.mean() for t in T])
        se = np.array([cm[cm.temp == t].NCI.std(ddof=1) / np.sqrt(90) for t in T])
        ax.fill_between(T, np.clip(mu - se, 0, 1), np.clip(mu + se, 0, 1), color=COL[m], alpha=0.22, lw=0, zorder=2)
        ax.plot(T, mu, color=COL[m], lw=2.0, marker=MK[m], ms=8, mec="white", mew=1.0, zorder=4,
                label=LAB[m], clip_on=False)
    ax.annotate("Claude: API maximum T = 1.0", xy=(1.0, 0.996), xytext=(1.06, 0.9952), fontsize=9.5,
                color="#555555", va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.8))
    ax.set_xlim(-0.04, 1.54); ax.set_ylim(0.915, 1.003)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5]); ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_xlabel("Sampling temperature"); ax.set_ylabel("Mean response consistency\n(1 − normalized entropy)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handletextpad=0.4,
              columnspacing=1.2)
    save(fig, "Figure1")

def fig2(cells, R):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 5.0), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.28))
    a.grid(color=GRID, lw=0.7); a.set_axisbelow(True)
    a.plot([0, 1], [0, 1], color=REF, lw=1.0, ls=(0, (4, 3)), label="Perfect calibration", zorder=1)
    for m in M:
        bins = [x for x in R["D_calibration"][m]["bins"] if x["n"]]
        a.plot([x["mean_conf"] for x in bins], [x["acc"] for x in bins], color=COL[m], lw=2.0, marker=MK[m], ms=8,
               mec="white", mew=1.0, zorder=3, label=f"{LAB[m]}  (ECE {R['D_calibration'][m]['ECE']:.3f})")
    a.set_xlim(0.45, 1.02); a.set_ylim(-0.03, 1.04)
    a.xaxis.set_major_locator(MultipleLocator(0.1)); a.yaxis.set_major_locator(MultipleLocator(0.2))
    a.set_xlabel("Modal-answer frequency (sampling-based confidence score)")
    a.set_ylabel("Observed accuracy of modal answer")
    a.legend(loc="lower right", frameon=False)
    a.text(-0.13, 1.03, "a", transform=a.transAxes, fontsize=15, fontweight="bold")
    b.grid(axis="y", color=GRID, lw=0.7); b.set_axisbelow(True)
    xs = np.arange(len(M)); n = [len(R["B_residual"][m]) for m in M]; pct = [100 * k / 90 for k in n]
    b.bar(xs, pct, width=0.56, color=[COL[m] for m in M], edgecolor="white", linewidth=2, zorder=3)
    for x, p, k in zip(xs, pct, n):
        b.text(x, p + 0.15, f"{p:.1f}%\n({k}/90)", ha="center", va="bottom", fontsize=11, fontweight="bold",
               color="#111111")
    b.set_xticks(xs); b.set_xticklabels(["GPT-4o", "Claude\nOpus 4.5", "Gemini 2.5\nFlash-Lite"])
    b.set_ylim(0, 9.5); b.yaxis.set_major_locator(MultipleLocator(2))
    b.set_ylabel("Consistently wrong items (% of 90)")
    b.text(-0.13, 1.03, "b", transform=b.transAxes, fontsize=15, fontweight="bold")
    save(fig, "Figure2")

def fig3(cells, R, bands):
    fig, ax = plt.subplots(figsize=(6.9, 5.4))
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color=REF, lw=1.0, ls=(0, (4, 3)), label="Random escalation", zorder=1)
    for m in M:
        lo, hi = bands[m]; avg = model_curves(m, cells)
        ax.fill_between(GRIDX, lo, hi, color=COL[m], alpha=0.20, lw=0, zorder=2)
        s = R["E_triage"]["summary"][m]
        ax.plot(GRIDX, avg, color=COL[m], lw=2.0, zorder=3,
                label=f"{LAB[m]}  (AUC {s['expected_mean']:.2f}, 95% CI {s['boot_lo']:.2f}–{s['boot_hi']:.2f})")
        idx = np.array([8, 36, 64, 92])
        ax.plot(GRIDX[idx], avg[idx], ls="none", marker=MK[m], ms=8, color=COL[m], mec="white", mew=1.0, zorder=4)
    # legend: proxy handles (line + marker) so the plotted curves are not modified
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    proxies = [h[0]] + [Line2D([], [], color=COL[m], lw=2.0, marker=MK[m], ms=7, mec="white", mew=1.0) for m in M]
    ax.legend(proxies, l, loc="lower right", frameon=False, fontsize=9.6, handlelength=2.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.2)); ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlabel("Fraction of items escalated to human review\n(lowest consistency first; ties in random order)")
    ax.set_ylabel("Expected fraction of model's errors caught")
    save(fig, "Figure3")

if __name__ == "__main__":
    cells = pd.read_pickle(ra.OUT / "cells.pkl")
    R = json.load(open(ra.OUT / "revision_results.json"))
    print("[1/4] bootstrap bands (parallel, cached)", flush=True)
    with Pool(2) as p:
        res = p.map(boot_band, M)
    bands = dict(zip(M, res))
    print("[2/4] Figure 1", flush=True); fig1(cells)
    print("[3/4] Figure 2", flush=True); fig2(cells, R)
    print("[4/4] Figure 3", flush=True); fig3(cells, R, bands)
    print("done")
