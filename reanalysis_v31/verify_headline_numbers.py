#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_headline_numbers.py — Reproduce every headline Study-2 statistic in the
manuscript directly from the public raw responses (raw_dedup_29700.csv),
using the canonical reanalysis convention (valid = non-empty parsed letter).

Run:  python3 verify_headline_numbers.py  path/to/raw_dedup_29700.csv  path/to/exp5_prompts.csv
Prints ECE, unanimous-but-wrong residual, triage AUC, and Spearman rho, and
checks them against the values reported in the paper.

Revision note (October 2026, ACI-2026-07-RA-0313 R1): the triage AUC is now computed with the tie-corrected
expected capture curve (random ordering within tied consistency values, equivalent to counting ties as one half).
The originally submitted values (0.71 / 0.52 / 0.75) came from an arbitrary order of tied items; they are still
reproduced below for transparency but are superseded. Full revision analyses: ../revision_r1/.
"""
import sys, math
import numpy as np, pandas as pd
from scipy import stats

from pathlib import Path
_R = Path(__file__).resolve().parent.parent  # repository root
RAW = sys.argv[1] if len(sys.argv) > 1 else str(_R / "study2_japanese_national_exam/exp5_results/raw_dedup_29700.csv")
PROMPTS = sys.argv[2] if len(sys.argv) > 2 else str(_R / "study2_japanese_national_exam/exp5_pool/exp5_prompts.csv")
MODELS = ["gpt-4o", "claude-opus-4.5", "gemini-2.5-flash-lite"]
LAB = {"gpt-4o": "GPT-4o", "claude-opus-4.5": "Claude", "gemini-2.5-flash-lite": "Gemini"}
REPORTED = {  # values reported in the revised manuscript (R1)
    "ECE": {"GPT-4o": 0.039, "Claude": 0.023, "Gemini": 0.059},
    "residual_n": {"GPT-4o": 4, "Claude": 2, "Gemini": 6},
    "AUC": {"GPT-4o": 0.77, "Claude": 0.50, "Gemini": 0.68},              # tie-corrected (revised)
    "AUC_arbitrary_ties": {"GPT-4o": 0.71, "Claude": 0.52, "Gemini": 0.75},  # originally submitted, superseded
    "rho": 0.956,
}

def main():
    raw = pd.read_csv(RAW, encoding="utf-8-sig")
    pr = pd.read_csv(PROMPTS, encoding="utf-8-sig")
    kmap = dict(zip(pr.item_uid, pr.k))
    raw["k"] = raw.item_uid.map(kmap)
    # CANONICAL CONVENTION: valid = non-empty parsed letter (empty responses excluded)
    raw["valid"] = raw.parsed_letter.notna() & (raw.parsed_letter.astype(str).str.strip() != "")
    n_empty = int((~raw.valid).sum())
    print(f"raw calls: {len(raw)}  |  empty/excluded: {n_empty} ({100*n_empty/len(raw):.2f}%)")

    # per-cell metrics over valid responses
    rows = []
    for (it, m, t), g in raw[raw.valid].groupby(["item_uid", "model", "temp"]):
        vc = g.parsed_letter.value_counts(); n = int(vc.sum()); k = int(kmap[it])
        H = -sum((c/n)*math.log2(c/n) for c in vc.values if c > 0)
        nci = 1 - H/math.log2(k)
        modal = vc.idxmax(); phat = int(vc.max())/n
        key = g.key_letter.iloc[0]
        rows.append(dict(item=it, model=m, temp=t, NCI=nci, phat=phat,
                         accuracy=float((g.parsed_letter == key).mean()),
                         modal_correct=int(modal == key),
                         category=g.category.iloc[0]))
    cells = pd.DataFrame(rows)

    def ece(conf, correct, bins=10):  # (lo, hi] right-closed bins
        conf = np.asarray(conf, float); correct = np.asarray(correct, float)
        edges = np.linspace(0, 1, bins+1); N = len(conf); e = 0.0
        for i in range(bins):
            lo, hi = edges[i], edges[i+1]
            m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            if m.sum() == 0: continue
            e += (m.sum()/N)*abs(correct[m].mean() - conf[m].mean())
        return e

    def triage_auc(sub):  # originally submitted version: arbitrary order of tied items (superseded)
        s = sub.sort_values("NCI").reset_index(drop=True); n = len(s)
        w = int((s.modal_correct == 0).sum())
        if w == 0: return None
        cap = np.cumsum((s.modal_correct == 0).values)/w; frac = np.arange(1, n+1)/n
        return float(np.trapezoid(cap, frac))

    def triage_auc_tie_corrected(sub):  # revised: expected capture curve under random order within ties
        nci = np.round(sub.NCI.values.astype(float), 12); err = (sub.modal_correct.values == 0).astype(int)
        W = int(err.sum()); n = len(err)
        if W == 0: return None
        o = np.argsort(nci, kind="mergesort"); v = nci[o]; e = err[o]
        starts = np.r_[0, np.flatnonzero(np.diff(v) != 0) + 1]; ends = np.r_[starts[1:], n]
        caught = np.zeros(n); before = 0.0
        for a, b in zip(starts, ends):
            size = b - a; eg = int(e[a:b].sum())
            caught[a:b] = before + eg * np.arange(1, size + 1) / size; before += eg
        return float(np.trapezoid(caught / W, np.arange(1, n + 1) / n))

    ok = True
    print("\n== ECE (10 bins, right-closed) ==")
    for m in MODELS:
        c = cells[cells.model == m]; e = ece(c.phat, c.modal_correct)
        r = REPORTED["ECE"][LAB[m]]; hit = round(e, 3) == r; ok &= hit
        print(f"  {LAB[m]:7}: {e:.3f}  (paper {r})  {'OK' if hit else 'MISMATCH'}")

    print("\n== unanimous-but-wrong residual (items with any temperature unanimous & wrong) ==")
    for m in MODELS:
        byit = cells[cells.model == m].groupby("item")
        n = sum(1 for _, g in byit if ((g.phat >= 0.999) & (g.modal_correct == 0)).any())
        r = REPORTED["residual_n"][LAB[m]]; hit = n == r; ok &= hit
        print(f"  {LAB[m]:7}: {n}/90  (paper {r})  {'OK' if hit else 'MISMATCH'}")

    print("\n== triage AUC (temperature-averaged; tie-corrected, as revised) ==")
    for m in MODELS:
        temps = [0.0, 0.5, 1.0, 1.5] if m != "claude-opus-4.5" else [0.0, 0.5, 1.0]
        aucs = [triage_auc_tie_corrected(cells[(cells.model == m) & (cells.temp == t)]) for t in temps]
        aucs = [a for a in aucs if a is not None]; mA = float(np.mean(aucs))
        r = REPORTED["AUC"][LAB[m]]; hit = round(mA, 2) == r; ok &= hit
        print(f"  {LAB[m]:7}: {mA:.2f}  (paper {r})  {'OK' if hit else 'MISMATCH'}")
    print("  (originally submitted values, arbitrary order of tied items; superseded:)")
    for m in MODELS:
        temps = [0.0, 0.5, 1.0, 1.5] if m != "claude-opus-4.5" else [0.0, 0.5, 1.0]
        aucs = [triage_auc(cells[(cells.model == m) & (cells.temp == t)]) for t in temps]
        aucs = [a for a in aucs if a is not None]; mA = float(np.mean(aucs))
        r = REPORTED["AUC_arbitrary_ties"][LAB[m]]; hit = round(mA, 2) == r; ok &= hit
        print(f"  {LAB[m]:7}: {mA:.2f}  (submitted {r})  {'OK' if hit else 'MISMATCH'}")

    item = cells.groupby("item").agg(acc=("accuracy", "mean"), nci=("NCI", "mean"))
    rho, p = stats.spearmanr(item.acc, item.nci)
    hit = round(rho, 3) == REPORTED["rho"]; ok &= hit
    print(f"\n== Spearman rho (item-level, n=90) ==\n  {rho:.3f}  p={p:.3g}  (paper {REPORTED['rho']})  {'OK' if hit else 'MISMATCH'}")

    print(f"\n{'ALL HEADLINE NUMBERS REPRODUCED' if ok else 'SOME MISMATCH — CHECK ABOVE'}")

if __name__ == "__main__":
    main()
