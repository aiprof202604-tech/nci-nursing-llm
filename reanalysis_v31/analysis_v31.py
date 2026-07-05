#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCI v31 reframe — three-layer reanalysis on EXISTING raw data (no new API calls).

Layer 1  Calibration : self-consistency confidence (modal-answer frequency) vs
                       observed accuracy; reliability, ECE, high-confidence error,
                       the "confidently wrong" quadrant.
Layer 2  Moderators  : what predicts item-level (in)consistency — temperature,
                       judgement category, language/study, item difficulty, k, recency.
Layer 3  Triage      : escalate low-consistency items to human review; error-capture
                       vs review-load, AUC, and review budget to catch X% of errors,
                       benchmarked against a random-triage baseline.

British English throughout. Idempotent; prints progress. Outputs -> OUT_DIR.
"""

import os, sys, math, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

REPO = str(Path(__file__).resolve().parent.parent)  # repository root (this file lives in reanalysis_v31/)
OUT_DIR = str(Path(__file__).resolve().parent / "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- house style (per author's palette) ----
COL = {"GPT-4o": "#3B6CA8", "Claude": "#C0504D", "Gemini": "#4F8A5B"}
MRK = {"GPT-4o": "o", "Claude": "s", "Gemini": "^"}
MODELS = ["GPT-4o", "Claude", "Gemini"]
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 140, "savefig.bbox": "tight"})

def log(m): print(f"[v31] {m}", flush=True)

# ----------------------------------------------------------------------
# 1. LOAD + HARMONISE both studies to a common per-trial schema
#    columns: study, model, temp, item, k, category, recent, resp, key, valid
# ----------------------------------------------------------------------
def load():
    log("loading Study 1 (English, 30 items) ...")
    s1 = pd.read_csv(f"{REPO}/data/raw_responses.csv")
    m1 = {"gpt": "GPT-4o", "claude": "Claude", "gemini": "Gemini"}
    s1 = pd.DataFrame({
        "study": "S1-EN", "model": s1.model.map(m1), "temp": s1.temperature.astype(float),
        "item": s1.question_id.astype(str), "k": 4, "category": s1.category.astype(str),
        "recent": "NA",
        "resp": s1.response.astype(str), "key": s1.intended_answer.astype(str),
        "valid": s1.is_valid.astype(bool),
    })

    log("loading Study 2 (Japanese national exam, 90 items) ...")
    s2 = pd.read_csv(f"{REPO}/study2_japanese_national_exam/exp5_results/raw_dedup_29700.csv")
    pr = pd.read_csv(f"{REPO}/study2_japanese_national_exam/exp5_pool/exp5_prompts.csv")
    kmap = dict(zip(pr.item_uid, pr.k)); rmap = dict(zip(pr.item_uid, pr.recent))
    m2 = {"gpt-4o": "GPT-4o", "claude-opus-4.5": "Claude", "gemini-2.5-flash-lite": "Gemini"}
    s2 = pd.DataFrame({
        "study": "S2-JP", "model": s2.model.map(m2), "temp": s2.temp.astype(float),
        "item": s2.item_uid.astype(str), "k": s2.item_uid.map(kmap).astype(int),
        "category": s2.category.replace({"EthicalLegal": "Ethical"}).astype(str),
        "recent": s2.item_uid.map(rmap).astype(str),
        "resp": s2.parsed_letter.astype(str), "key": s2.key_letter.astype(str),
        "valid": s2.parsed_letter.notna().values,
    })
    df = pd.concat([s1, s2], ignore_index=True)
    df.loc[~df.valid, "resp"] = np.nan
    log(f"combined per-trial rows: {len(df):,}  (valid {int(df.valid.sum()):,})")
    return df

# ----------------------------------------------------------------------
# 2. PER-CELL metrics: NCI, accuracy, modal answer, self-consistency conf
#    one row per (study, model, temp, item)
# ----------------------------------------------------------------------
def shannon(counts):
    n = counts.sum()
    if n == 0: return np.nan
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())

def per_cell(df, min_valid=10):
    log("computing per-cell NCI / accuracy / self-consistency confidence ...")
    rows = []
    grp = df[df.valid].groupby(["study", "model", "temp", "item", "k", "category", "recent"])
    for (study, model, temp, item, k, cat, recent), g in grp:
        vc = g.resp.value_counts()
        n = int(vc.sum())
        if n == 0: continue
        H = shannon(vc.values.astype(float))
        nci = 1.0 - H / math.log2(k) if k > 1 else np.nan
        modal = vc.idxmax(); modal_n = int(vc.max())
        phat = modal_n / n                      # self-consistency confidence
        key = g.key.iloc[0]
        acc = float((g.resp == key).mean())     # per-trial accuracy in the cell
        modal_correct = int(modal == key)       # majority-vote correctness
        rows.append(dict(study=study, model=model, temp=temp, item=item, k=k,
                         category=cat, recent=recent, n_valid=n,
                         NCI=nci, accuracy=acc, modal=modal, phat=phat,
                         modal_correct=modal_correct, key=key))
    cells = pd.DataFrame(rows)
    dropped = int((cells.n_valid < min_valid).sum())
    cells = cells[cells.n_valid >= min_valid].reset_index(drop=True)
    log(f"cells retained: {len(cells)} (dropped {dropped} with n_valid<{min_valid})")
    return cells

# ----------------------------------------------------------------------
# LAYER 1 — CALIBRATION
# ----------------------------------------------------------------------
def ece(conf, correct, bins=10):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1); N = len(conf); e = 0.0; table = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            table.append((lo, hi, 0, np.nan, np.nan)); continue
        c, a = conf[m].mean(), correct[m].mean()
        e += (m.sum() / N) * abs(a - c)
        table.append((lo, hi, int(m.sum()), float(c), float(a)))
    return float(e), pd.DataFrame(table, columns=["lo", "hi", "n", "conf", "acc"])

def layer1_calibration(cells):
    log("LAYER 1 — calibration (self-consistency confidence vs accuracy) ...")
    out = []
    for model in MODELS:
        c = cells[cells.model == model]
        E, _ = ece(c.phat, c.modal_correct, bins=10)
        # coarse, interpretable consistency bands
        band = pd.cut(c.phat, [0, .5, .8, .999, 1.0001],
                      labels=["low(<.5)", "med(.5-.8)", "high(.8-1)", "unanimous(=1)"])
        bt = c.groupby(band).modal_correct.agg(["size", "mean"])
        hi = c[c.phat >= 0.8]
        unanimous = c[c.phat >= 0.999]
        out.append(dict(
            model=model, n_cells=len(c), ECE=E,
            acc_when_high_consistency=float(hi.modal_correct.mean()) if len(hi) else np.nan,
            err_rate_high_consistency=float(1 - hi.modal_correct.mean()) if len(hi) else np.nan,
            confidently_wrong_share=float(((c.phat >= 0.8) & (c.modal_correct == 0)).mean()),
            unanimous_but_wrong_share=float(((c.phat >= 0.999) & (c.modal_correct == 0)).mean()),
            n_unanimous=len(unanimous),
            err_rate_unanimous=float(1 - unanimous.modal_correct.mean()) if len(unanimous) else np.nan,
        ))
        # per-band table to disk
        bt.to_csv(f"{OUT_DIR}/L1_bands_{model}.csv")
    res = pd.DataFrame(out)
    res.to_csv(f"{OUT_DIR}/L1_calibration_summary.csv", index=False)
    return res

# ----------------------------------------------------------------------
# LAYER 2 — MODERATORS
# ----------------------------------------------------------------------
def ols_r2(y, X):
    X = np.column_stack([np.ones(len(X)), X]); 
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta; ss_res = ((y - yhat) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    return beta, 1 - ss_res / ss_tot

def layer2_moderators(cells):
    log("LAYER 2 — moderators of (in)consistency ...")
    # group means
    by_cat = cells.groupby(["model", "category"]).agg(
        NCI=("NCI", "mean"), accuracy=("accuracy", "mean"), n=("NCI", "size")).reset_index()
    by_temp = cells.groupby(["model", "temp"]).agg(
        NCI=("NCI", "mean"), accuracy=("accuracy", "mean"), n=("NCI", "size")).reset_index()
    by_study = cells.groupby(["model", "study"]).agg(
        NCI=("NCI", "mean"), accuracy=("accuracy", "mean"), n=("NCI", "size")).reset_index()
    by_cat.to_csv(f"{OUT_DIR}/L2_by_category.csv", index=False)
    by_temp.to_csv(f"{OUT_DIR}/L2_by_temperature.csv", index=False)
    by_study.to_csv(f"{OUT_DIR}/L2_by_study.csv", index=False)

    # item-level difficulty (pooled) vs item-level consistency
    item = cells.groupby(["study", "item"]).agg(
        item_acc=("accuracy", "mean"), item_NCI=("NCI", "mean")).reset_index()
    rho, p = stats.spearmanr(item.item_acc, item.item_NCI)

    # tests + simple model
    tests = {}
    tests["spearman_itemAcc_itemNCI_rho"] = float(rho)
    tests["spearman_itemAcc_itemNCI_p"] = float(p)
    # temperature effect (Spearman, per model)
    for model in MODELS:
        c = cells[cells.model == model]
        r, pp = stats.spearmanr(c.temp, c.NCI)
        tests[f"temp_vs_NCI_rho[{model}]"] = float(r)
        tests[f"temp_vs_NCI_p[{model}]"] = float(pp)
    # category effect (Kruskal-Wallis on NCI, pooled)
    groups = [g.NCI.values for _, g in cells.groupby("category")]
    H, pk = stats.kruskal(*groups)
    tests["category_KW_H"] = float(H); tests["category_KW_p"] = float(pk)
    # language effect (Mann-Whitney on NCI, S1 vs S2)
    U, pu = stats.mannwhitneyu(cells[cells.study == "S1-EN"].NCI,
                               cells[cells.study == "S2-JP"].NCI, alternative="two-sided")
    tests["language_MWU_p"] = float(pu)
    # OLS: NCI ~ temp + item_acc(difficulty proxy) + study(0/1)
    d = cells.merge(item.rename(columns={"item_acc": "item_acc"}), on=["study", "item"], how="left")
    X = np.column_stack([d.temp.values, d.item_acc.values, (d.study == "S2-JP").astype(float).values])
    beta, r2 = ols_r2(d.NCI.values, X)
    tests["OLS_intercept"] = float(beta[0]); tests["OLS_b_temp"] = float(beta[1])
    tests["OLS_b_itemAcc"] = float(beta[2]); tests["OLS_b_studyJP"] = float(beta[3])
    tests["OLS_R2"] = float(r2)
    # recency (S2 only): recent vs older
    s2 = cells[cells.study == "S2-JP"]
    if s2.recent.nunique() > 1:
        U2, pr2 = stats.mannwhitneyu(s2[s2.recent == "Y"].NCI, s2[s2.recent == "N"].NCI,
                                     alternative="two-sided")
        tests["recencyY_meanNCI"] = float(s2[s2.recent == "Y"].NCI.mean())
        tests["recencyN_meanNCI"] = float(s2[s2.recent == "N"].NCI.mean())
        tests["recency_MWU_p"] = float(pr2)
    json.dump(tests, open(f"{OUT_DIR}/L2_tests.json", "w"), ensure_ascii=False, indent=2)
    item.to_csv(f"{OUT_DIR}/L2_item_level.csv", index=False)
    return by_cat, by_temp, by_study, item, tests

# ----------------------------------------------------------------------
# LAYER 3 — TRIAGE (consistency-based escalation)
# ----------------------------------------------------------------------
def triage_curve(sub, grid):
    """sub: rows with NCI + modal_correct for one (model,temp). Rank by NCI asc."""
    s = sub.sort_values("NCI", ascending=True).reset_index(drop=True)
    n = len(s); wrong_total = int((s.modal_correct == 0).sum())
    if wrong_total == 0: return None
    caught = np.cumsum((s.modal_correct == 0).values)
    frac_reviewed = (np.arange(1, n + 1)) / n
    cap = caught / wrong_total
    # interpolate capture onto common review-fraction grid
    cg = np.interp(grid, np.concatenate([[0], frac_reviewed]),
                   np.concatenate([[0], cap]))
    auc = float(np.trapezoid(cap, frac_reviewed))            # area under capture curve
    # review load to catch 80% / 90%
    def load_for(t):
        idx = np.searchsorted(cap, t)
        return float(frac_reviewed[min(idx, n - 1)])
    return dict(auc=auc, base_err=wrong_total / n,
                load80=load_for(.8), load90=load_for(.9), curve=cg)

def layer3_triage(cells):
    log("LAYER 3 — consistency-based triage (per model x temp; isolates item-level signal) ...")
    grid = np.linspace(0, 1, 101)
    rows = []; curves = {m: [] for m in MODELS}
    for (model, temp), sub in cells.groupby(["model", "temp"]):
        if len(sub) < 20: continue
        r = triage_curve(sub, grid)
        if r is None: continue
        rows.append(dict(model=model, temp=temp, n_items=len(sub),
                         base_err=r["base_err"], AUC=r["auc"],
                         load_for_80pct=r["load80"], load_for_90pct=r["load90"]))
        curves[model].append(r["curve"])
    res = pd.DataFrame(rows).sort_values(["model", "temp"])
    res.to_csv(f"{OUT_DIR}/L3_triage_by_model_temp.csv", index=False)
    # temp-averaged curve + summary per model
    summ = []; avg_curves = {}
    for model in MODELS:
        if not curves[model]: continue
        avg = np.mean(np.vstack(curves[model]), axis=0); avg_curves[model] = avg
        sub = res[res.model == model]
        summ.append(dict(model=model, mean_AUC=float(sub.AUC.mean()),
                         mean_base_err=float(sub.base_err.mean()),
                         mean_load_for_80pct=float(sub.load_for_80pct.mean()),
                         mean_load_for_90pct=float(sub.load_for_90pct.mean()),
                         random_AUC=0.5))
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{OUT_DIR}/L3_triage_summary.csv", index=False)
    return res, summ, grid, avg_curves

# ----------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------
def figures(cells, l1, by_temp, item, grid, avg_curves):
    log("rendering figures ...")
    # Fig A — reliability diagram (calibration)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for model in MODELS:
        c = cells[cells.model == model]
        _, tb = ece(c.phat, c.modal_correct, bins=10)
        tb = tb.dropna()
        ax.plot(tb.conf, tb.acc, marker=MRK[model], color=COL[model], lw=1.8,
                ms=7, label=f"{model} (ECE={l1.set_index('model').loc[model,'ECE']:.3f})")
    ax.set_xlabel("Self-consistency confidence (modal-answer frequency)")
    ax.set_ylabel("Observed accuracy of the majority vote")
    ax.set_title("Layer 1 — Calibration of self-consistency confidence")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.legend(fontsize=8.5, loc="upper left")
    fig.savefig(f"{OUT_DIR}/FigA_calibration.png"); plt.close(fig)

    # Fig B — accuracy vs NCI scatter, dangerous quadrant
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for model in MODELS:
        c = cells[cells.model == model]
        ax.scatter(c.NCI, c.accuracy, s=14, alpha=.35, color=COL[model],
                   marker=MRK[model], edgecolors="none", label=model)
    ax.axhline(.5, color="grey", lw=.8, ls=":"); ax.axvline(.8, color="grey", lw=.8, ls=":")
    ax.text(.985, .03, "high consistency,\nlow accuracy\n= 'confidently wrong'",
            ha="right", va="bottom", fontsize=8.5, color="#7a1f1c",
            bbox=dict(boxstyle="round,pad=.3", fc="#fbeaea", ec="#c0504d", alpha=.9))
    ax.set_xlabel("NCI  (per-item response consistency)")
    ax.set_ylabel("Per-item accuracy")
    ax.set_title("Layer 1 — Consistency is not correctness (cell level)")
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02); ax.legend(fontsize=9)
    fig.savefig(f"{OUT_DIR}/FigB_consistency_vs_correctness.png"); plt.close(fig)

    # Fig C — triage capture curves
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random triage")
    for model in MODELS:
        if model in avg_curves:
            ax.plot(grid, avg_curves[model], color=COL[model], lw=2,
                    marker=MRK[model], markevery=12, ms=6, label=f"{model} (NCI-ranked)")
    ax.set_xlabel("Fraction of items escalated to human review")
    ax.set_ylabel("Fraction of model errors caught")
    ax.set_title("Layer 3 — Consistency-based triage (temp-averaged)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(fontsize=9, loc="lower right")
    fig.savefig(f"{OUT_DIR}/FigC_triage.png"); plt.close(fig)

    # Fig D — moderator: NCI by temperature
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for model in MODELS:
        t = by_temp[by_temp.model == model].sort_values("temp")
        ax.plot(t.temp, t.NCI, marker=MRK[model], color=COL[model], lw=1.8, ms=7, label=model)
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel("Mean NCI")
    ax.set_title("Layer 2 — Temperature as a moderator of consistency")
    ax.legend(fontsize=9)
    fig.savefig(f"{OUT_DIR}/FigD_temperature.png"); plt.close(fig)
    log("figures written.")

# ----------------------------------------------------------------------
def main():
    df = load()
    cells = per_cell(df)
    cells.to_csv(f"{OUT_DIR}/cells_all.csv", index=False)
    l1 = layer1_calibration(cells)
    by_cat, by_temp, by_study, item, tests = layer2_moderators(cells)
    l3_bymt, l3_summ, grid, avg_curves = layer3_triage(cells)
    figures(cells, l1, by_temp, item, grid, avg_curves)

    log("\n================  HEADLINE RESULTS  ================")
    log("LAYER 1 calibration:\n" + l1.to_string(index=False))
    log("\nLAYER 3 triage (temp-averaged):\n" + l3_summ.to_string(index=False))
    log("\nLAYER 2 key tests:")
    for k in ["spearman_itemAcc_itemNCI_rho", "spearman_itemAcc_itemNCI_p",
              "category_KW_p", "language_MWU_p", "OLS_b_temp", "OLS_b_itemAcc", "OLS_R2"]:
        log(f"   {k}: {tests[k]:.4g}")
    log("by study (NCI / accuracy):\n" + by_study.to_string(index=False))
    log("\nall outputs -> " + OUT_DIR)

if __name__ == "__main__":
    main()
