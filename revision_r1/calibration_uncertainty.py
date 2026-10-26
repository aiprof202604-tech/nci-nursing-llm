#!/usr/bin/env python3
"""Uncertainty of the calibration estimates (Supplementary S10.2, Table S8b).
Item-cluster bootstrap: 2,000 resamples of the 90 items with all their temperature cells (same unit and resampling
scheme as the correlation analysis). For each model and resample: ECE (10 equal-width, right-closed bins, first bin
closed), Brier score of the modal-answer frequency against modal-answer correctness, the Brier score of a constant
forecast at the resample's base rate, their difference (negative = better than the base rate), and the resolution
component. Percentile 95% intervals. Runs in parallel chunks cached in outputs/cache/ (resumable); prints progress.
Inputs: the public raw data (NCI_REPO). No API calls. Output: outputs/calibration_uncertainty.json."""
import json, math, os, time
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(os.environ.get("NCI_REPO", "nci-nursing-llm")).resolve()
OUT = Path(os.environ.get("NCI_OUT", "outputs")).resolve()
CACHE = OUT / "cache"
RAW = REPO / "study2_japanese_national_exam" / "exp5_results" / "raw_dedup_29700.csv"
PROMPTS = REPO / "study2_japanese_national_exam" / "exp5_pool" / "exp5_prompts.csv"
MODELS = ["gpt-4o", "claude-opus-4.5", "gemini-2.5-flash-lite"]
B, NCHUNK, SEED = 2000, 8, 20261026
EDGES = np.linspace(0, 1, 11)


def build_cells():
    raw = pd.read_csv(RAW, encoding="utf-8-sig")
    pr = pd.read_csv(PROMPTS, encoding="utf-8-sig")
    kmap = dict(zip(pr.item_uid, pr.k))
    raw["valid"] = raw.parsed_letter.notna() & (raw.parsed_letter.astype(str).str.strip() != "")
    rows = []
    for (it, m, t), g in raw[raw.valid].groupby(["item_uid", "model", "temp"]):
        vc = g.parsed_letter.value_counts(); n = int(vc.sum())
        rows.append(dict(item=it, model=m, temp=float(t), phat=int(vc.max()) / n,
                         modal_correct=int(vc.idxmax() == g.key_letter.iloc[0])))
    return pd.DataFrame(rows)


def metrics(conf, y):
    N = len(conf); ece = 0.0; res = 0.0; ybar = y.mean()
    for i in range(10):
        lo, hi = EDGES[i], EDGES[i + 1]
        msk = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        nk = msk.sum()
        if nk:
            ok = y[msk].mean()
            ece += nk / N * abs(ok - conf[msk].mean()); res += nk * (ok - ybar) ** 2
    brier = float(np.mean((conf - y) ** 2)); ref = float(ybar * (1 - ybar))
    return dict(ECE=float(ece), Brier=brier, Brier_ref=ref, Brier_minus_ref=brier - ref, resolution=float(res / N))


def chunk(args):
    model, c, n, seed = args
    f = CACHE / f"calib_{model}_{c}.json"
    if f.exists():
        return json.loads(f.read_text())                      # resume
    cells = pd.read_pickle(OUT / "calib_cells.pkl")
    cm = cells[cells.model == model]
    items = np.array(sorted(cm.item.unique()))
    groups = {it: g for it, g in cm.groupby("item")}
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        d = pd.concat([groups[it] for it in items[rng.integers(0, len(items), len(items))]])
        out.append(metrics(d.phat.to_numpy(), d.modal_correct.to_numpy()))
    f.write_text(json.dumps(out))
    return out


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    print("building cells ...", flush=True)
    cells = build_cells(); cells.to_pickle(OUT / "calib_cells.pkl")
    per = B // NCHUNK
    tasks = [(m, c, per, SEED + 100 * i + c) for i, m in enumerate(MODELS) for c in range(NCHUNK)]
    res = {}; t0 = time.time()
    with Pool(min(len(tasks), cpu_count())) as p:
        for k, (args, r) in enumerate(zip(tasks, p.imap(chunk, tasks)), 1):
            res.setdefault(args[0], []).extend(r)
            print(f"\r  progress {k}/{len(tasks)} ({100 * k / len(tasks):.0f}%)  {time.time() - t0:.0f}s", end="", flush=True)
    print()
    R = {}
    for m in MODELS:
        cm = cells[cells.model == m]
        point = metrics(cm.phat.to_numpy(), cm.modal_correct.to_numpy())
        R[m] = dict(n_cells=len(cm), n_items=int(cm.item.nunique()), point=point, boot_n=len(res[m]))
        for key in point:
            v = np.array([r[key] for r in res[m]])
            R[m][key + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        print(m, {k: round(v, 4) for k, v in point.items()},
              {k: [round(x, 4) for x in R[m][k + "_ci"]] for k in point})
    (OUT / "calibration_uncertainty.json").write_text(json.dumps(R, indent=1))
    print("written", OUT / "calibration_uncertainty.json")


if __name__ == "__main__":
    main()
