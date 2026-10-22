#!/usr/bin/env python3
"""Cells whose modal answer is tied (two or more answers share the maximum count).

The analysis code takes the modal answer with pandas value_counts().idxmax(), which resolves a tie in favour of the
answer that occurs first in the data. This script lists every tied cell, shows the answer of its first trial, and
re-estimates triage AUC (point estimate and the joint item bootstrap, same seeds as revision_analyses.py), ECE, Brier
score, and the consistently-wrong counts under the alternative rule "a tie that includes the key counts as correct".
Reads only the public raw data (via revision_analyses.py paths); no API calls."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import revision_analyses as ra

def ece_brier(conf, y):
    """ECE (10 equal-width right-closed bins) and Brier score with its binned Murphy decomposition."""
    edges = np.linspace(0, 1, 11); ece = rel = res = 0.0; N = len(conf); ybar = y.mean()
    for k in range(10):
        lo, hi = edges[k], edges[k + 1]
        msk = (conf > lo) & (conf <= hi) if k else (conf >= lo) & (conf <= hi)
        nk = int(msk.sum())
        if nk:
            ck, ok_ = conf[msk].mean(), y[msk].mean()
            ece += nk / N * abs(ok_ - ck); rel += nk * (ck - ok_) ** 2; res += nk * (ok_ - ybar) ** 2
    return float(ece), float(np.mean((conf - y) ** 2)), rel / N, res / N, float(ybar * (1 - ybar))

def boot_ci(cm, model_index, temps):
    """Same procedure and seeds as revision_analyses.py (8 chunks x 250 joint item resamples)."""
    items = np.array(sorted(cm["item"].unique()))
    by_t = {t: cm[cm.temp == t].set_index("item").loc[items] for t in temps}
    means = []; nch = 8; per = ra.B // nch; t0 = time.time()
    for ch in range(nch):
        rng = np.random.default_rng(ra.SEED + 50000 + 100 * model_index + ch)
        for _ in range(per):
            idx = rng.integers(0, len(items), len(items)); vals = []
            for t in temps:
                d = by_t[t].iloc[idx]
                a = ra.capture_auc(d.NCI.values, (d.modal_correct.values == 0).astype(int), "expected")
                if a is not None: vals.append(a)
            means.append(float(np.mean(vals)) if vals else None)
        print(f"\r    bootstrap chunk {ch + 1}/{nch}  {time.time() - t0:.0f}s", end="", flush=True)
    print()
    lo, hi, nv = ra.pct_ci(means)
    return round(lo, 4), round(hi, 4), nv

def main():
    raw = pd.read_csv(ra.RAW_DEDUP, encoding="utf-8-sig")
    raw["valid"] = raw.parsed_letter.notna() & (raw.parsed_letter.astype(str).str.strip() != "")
    cells = pd.read_pickle(ra.OUT / "cells.pkl")
    ties = []
    for (it, m, t), g in raw[raw.valid].groupby(["item_uid", "model", "temp"]):
        vc = g.parsed_letter.value_counts()
        top = vc[vc == vc.max()]
        if len(top) > 1:
            key = g.key_letter.iloc[0]
            first_trial = g.sort_values("trial").parsed_letter.iloc[0]
            ties.append(dict(item=it, model=m, temp=float(t), counts={k: int(v) for k, v in vc.items()},
                             tied=list(top.index), key=key, chosen=vc.idxmax(), first_trial_answer=first_trial,
                             key_in_tie=bool(key in list(top.index))))
    print("tied-mode cells:", len(ties))
    for x in ties: print("  ", x)

    alt = cells.copy()
    for x in ties:
        if x["key_in_tie"]:
            msk = (alt["item"] == x["item"]) & (alt.model == x["model"]) & (alt.temp == x["temp"])
            alt.loc[msk, "modal_correct"] = 1
    affected = sorted({x["model"] for x in ties if x["key_in_tie"]})
    out = {}
    for name, cdf in [("canonical", cells), ("key_wins_ties", alt)]:
        res = {}
        for i, m in enumerate(ra.MODELS):
            cm = cdf[cdf.model == m]; per = {}
            for t in ra.TEMPS[m]:
                ct = cm[cm.temp == t]
                per[t] = ra.capture_auc(ct.NCI.values, (ct.modal_correct.values == 0).astype(int), "expected")
            vals = [v for v in per.values() if v is not None]
            ece, brier, rel, reso, unc = ece_brier(cm.phat.values, cm.modal_correct.values.astype(float))
            resid = sum(1 for _, g in cm.groupby("item") if ((g.phat >= 0.999) & (g.modal_correct == 0)).any())
            res[m] = dict(per_temp={str(k): (None if v is None else round(v, 4)) for k, v in per.items()},
                          mean_auc=round(float(np.mean(vals)), 4), ECE=round(ece, 6), Brier=round(brier, 6),
                          reliability=round(rel, 6), resolution=round(reso, 6), uncertainty=round(unc, 6),
                          residual=resid, modal_accuracy=round(float(cm.modal_correct.mean()), 4))
            if m in affected:
                print(f"  {name}: bootstrap for {ra.LAB[m]}")
                res[m]["boot_ci"] = boot_ci(cm, i, ra.TEMPS[m])
        out[name] = res
        print(f"\n== {name} ==")
        for m in ra.MODELS: print(f"  {ra.LAB[m]:7s}", res[m])
    json.dump(dict(ties=ties, sensitivity=out), open(ra.OUT / "modal_tie_check.json", "w"), indent=1, default=str)
    print("written", ra.OUT / "modal_tie_check.json")

if __name__ == "__main__":
    main()
