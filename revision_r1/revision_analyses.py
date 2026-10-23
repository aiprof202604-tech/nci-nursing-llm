#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision_analyses.py — ACI-2026-07-RA-0313, Round-1 revision analyses (Reviewer 3, AE).

Reads ONLY the public raw data of the repository (no new API calls). Writes results to OUT.
Features requested for all project code: resume (per-task cache), parallel (multiprocessing),
progress display.

Sections
  A  Data audit (duplicates in raw log, empty / out-of-range responses, category x stratum)
  B  Unanimous-but-wrong ("consistently wrong") residual items (for R2-5 and Suppl. table)
  C  Consistency-accuracy correlation: unit of analysis & dependence (R3-1)
  D  Calibration: ECE bins with counts, Brier score + Murphy decomposition (R3-4)
  E  Triage capture-curve AUC: tie handling, temperature aggregation, bootstrap (R3-2)
"""
import os, sys, json, math, time, hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(os.environ.get("NCI_REPO", "nci-nursing-llm")).resolve()  # path to a clone of aiprof202604-tech/nci-nursing-llm
RAW_DEDUP = REPO / "study2_japanese_national_exam/exp5_results/raw_dedup_29700.csv"
RAW_LOG = REPO / "study2_japanese_national_exam/exp5_results/raw.csv"
PROMPTS = REPO / "study2_japanese_national_exam/exp5_pool/exp5_prompts.csv"
OUT = Path(os.environ.get("NCI_OUT", "outputs")).resolve(); OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"; CACHE.mkdir(exist_ok=True)

MODELS = ["gpt-4o", "claude-opus-4.5", "gemini-2.5-flash-lite"]
LAB = {"gpt-4o": "GPT-4o", "claude-opus-4.5": "Claude", "gemini-2.5-flash-lite": "Gemini"}
TEMPS = {"gpt-4o": [0.0, 0.5, 1.0, 1.5], "claude-opus-4.5": [0.0, 0.5, 1.0],
         "gemini-2.5-flash-lite": [0.0, 0.5, 1.0, 1.5]}
B = 2000
SEED = 20261022

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------------------------------------------------------- cells (canonical convention)
def build_cells():
    raw = pd.read_csv(RAW_DEDUP, encoding="utf-8-sig")
    pr = pd.read_csv(PROMPTS, encoding="utf-8-sig")
    kmap = dict(zip(pr.item_uid, pr.k))
    raw["valid"] = raw.parsed_letter.notna() & (raw.parsed_letter.astype(str).str.strip() != "")
    rows = []
    for (it, m, t), g in raw[raw.valid].groupby(["item_uid", "model", "temp"]):
        vc = g.parsed_letter.value_counts(); n = int(vc.sum()); k = int(kmap[it])
        H = -sum((c/n)*math.log2(c/n) for c in vc.values if c > 0)
        key = g.key_letter.iloc[0]
        rows.append(dict(item=it, model=m, temp=float(t), NCI=1 - H/math.log2(k), phat=int(vc.max())/n,
                         n_valid=n, accuracy=float((g.parsed_letter == key).mean()),
                         modal=vc.idxmax(), modal_correct=int(vc.idxmax() == key), key=key,
                         category=g.category.iloc[0]))
    return raw, pr, pd.DataFrame(rows)

# ---------------------------------------------------------------- E: triage helpers
def _order_and_groups(nci, err):
    o = np.argsort(nci, kind="stable")
    v = np.round(nci[o], 12); e = err[o]
    # group boundaries of tied values
    starts = np.r_[0, np.flatnonzero(np.diff(v) != 0) + 1]
    ends = np.r_[starts[1:], len(v)]
    return e, starts, ends

def capture_auc(nci, err, mode="expected"):
    """Area under the capture curve, same formula as the original analysis:
       trapezoid of cap_i over frac_i = i/n, i = 1..n.  mode:
       'expected' = expectation under random ordering within tied consistency values
       'best'     = errors placed first within ties;  'worst' = errors placed last."""
    nci = np.asarray(nci, float); err = np.asarray(err, int)
    W = int(err.sum()); n = len(err)
    if W == 0: return None
    e, starts, ends = _order_and_groups(nci, err)
    caught = np.zeros(n); before = 0.0
    for s, t in zip(starts, ends):
        size = t - s; eg = int(e[s:t].sum()); j = np.arange(1, size + 1)
        if mode == "expected": caught[s:t] = before + eg * j / size
        elif mode == "best":   caught[s:t] = before + np.minimum(j, eg)
        elif mode == "worst":  caught[s:t] = before + np.maximum(0, j - (size - eg))
        before += eg
    cap = caught / W; frac = np.arange(1, n + 1) / n
    return float(np.trapezoid(cap, frac))

def legacy_auc(sub):
    """Exact re-implementation of the original code (pandas default sort; arbitrary tie order)."""
    s = sub.sort_values("NCI").reset_index(drop=True); n = len(s)
    w = int((s.modal_correct == 0).sum())
    if w == 0: return None
    cap = np.cumsum((s.modal_correct == 0).values) / w; frac = np.arange(1, n + 1) / n
    return float(np.trapezoid(cap, frac))

def roc_auc_ties_half(nci, err):
    """P(error item has LOWER consistency than a correct item), ties counted 1/2."""
    nci = np.asarray(nci, float); err = np.asarray(err, int)
    a = nci[err == 1]; b = nci[err == 0]
    if len(a) == 0 or len(b) == 0: return None
    gt = (b[None, :] > a[:, None]).sum(); eq = (b[None, :] == a[:, None]).sum()
    return float((gt + 0.5 * eq) / (len(a) * len(b)))

# ---------------------------------------------------------------- bootstrap workers (parallel + resume)
_CELLS = None
def _init(cells_path):
    global _CELLS
    _CELLS = pd.read_pickle(cells_path)

def _boot_triage_chunk(args):
    model, chunk, nrep, seed = args
    f = CACHE / f"triage_{model}_{chunk}.json"
    if f.exists(): return json.loads(f.read_text())          # resume
    rng = np.random.default_rng(seed)
    cm = _CELLS[_CELLS.model == model]
    items = np.array(sorted(cm["item"].unique()))
    by_t = {t: cm[cm.temp == t].set_index("item").loc[items] for t in TEMPS[model]}
    res = []
    for _ in range(nrep):
        idx = rng.integers(0, len(items), len(items))          # JOINT item resampling across temperatures
        per_t = {}
        for t in TEMPS[model]:
            d = by_t[t].iloc[idx]
            err = (d.modal_correct.values == 0).astype(int)
            per_t[str(t)] = capture_auc(d.NCI.values, err, "expected")
        valid = [v for v in per_t.values() if v is not None]
        res.append(dict(per_t=per_t, mean=(float(np.mean(valid)) if valid else None),
                        n_temps_skipped=sum(v is None for v in per_t.values())))
    f.write_text(json.dumps(res))
    return res

def _boot_legacy_chunk(args):
    """Attempted replication of the ORIGINAL CI procedure (independent per-temperature item resampling,
       legacy tie order).  Used only to explain the change in the response letter."""
    model, chunk, nrep, seed = args
    f = CACHE / f"legacy_{model}_{chunk}.json"
    if f.exists(): return json.loads(f.read_text())
    rng = np.random.default_rng(seed)
    cm = _CELLS[_CELLS.model == model]
    tabs = {t: cm[cm.temp == t].reset_index(drop=True) for t in TEMPS[model]}
    res = []
    for _ in range(nrep):
        vals = []
        for t in TEMPS[model]:
            sub = tabs[t]; idx = rng.integers(0, len(sub), len(sub))
            a = legacy_auc(sub.iloc[idx].reset_index(drop=True))
            if a is not None: vals.append(a)
        res.append(float(np.mean(vals)) if vals else None)
    f.write_text(json.dumps(res))
    return res

def _boot_corr_chunk(args):
    kind, chunk, nrep, seed = args
    f = CACHE / f"corr_{kind}_{chunk}.json"
    if f.exists(): return json.loads(f.read_text())
    rng = np.random.default_rng(seed)
    c = _CELLS if kind.startswith("pooled") else _CELLS[_CELLS.model == kind.split("|")[0].split(":")[1]]
    assert len(c) > 0, kind
    items = np.array(sorted(c["item"].unique()))
    groups = {it: g for it, g in c.groupby("item")}
    level = kind.split("|")[-1]   # 'item' or 'cell'
    out = []
    for _ in range(nrep):
        pick = items[rng.integers(0, len(items), len(items))]
        if level == "item":
            agg = [(groups[it].accuracy.mean(), groups[it].NCI.mean()) for it in pick]
            x, y = zip(*agg)
        else:  # cell level: keep every cell of each resampled item (item-cluster bootstrap)
            d = pd.concat([groups[it] for it in pick]); x, y = d.accuracy.values, d.NCI.values
        r = stats.spearmanr(x, y).statistic
        out.append(None if (r is None or np.isnan(r)) else float(r))
    f.write_text(json.dumps(out))
    return out

def run_parallel(func, tasks, label):
    log(f"{label}: {len(tasks)} chunks on {min(len(tasks), cpu_count())} processes")
    results = {}; done = 0; t0 = time.time()
    with Pool(min(len(tasks), cpu_count()), initializer=_init, initargs=(str(OUT / 'cells.pkl'),)) as p:
        for args, r in zip(tasks, p.imap(func, tasks)):
            results[args[:2]] = r; done += 1
            print(f"\r    progress {done}/{len(tasks)} ({100*done/len(tasks):.0f}%)  {time.time()-t0:.0f}s", end="", flush=True)
    print(flush=True)
    return results

def pct_ci(vals):
    v = np.array([x for x in vals if x is not None], float)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

# ---------------------------------------------------------------- main
def main():
    R = {}
    log("building cells from raw_dedup_29700.csv ...")
    raw, pr, cells = build_cells()
    cells.to_pickle(OUT / "cells.pkl"); cells.to_csv(OUT / "cells_revision.csv", index=False)

    # ===================== A. audit
    log("A. data audit")
    A = {}
    A["n_calls"] = int(len(raw)); A["n_empty"] = int((~raw.valid).sum())
    A["empty_by_model_category"] = raw[~raw.valid].groupby(["model", "category"]).size().to_dict()
    A["empty_by_model_category"] = {f"{k[0]}|{k[1]}": int(v) for k, v in A["empty_by_model_category"].items()}
    kmap = dict(zip(pr.item_uid, pr.k))
    rv = raw[raw.valid].copy()
    rv["opt_index"] = rv.parsed_letter.astype(str).str.strip().str[0].map(lambda ch: ord(ch) - 64)
    rv["k"] = rv.item_uid.map(kmap)
    oor = rv[rv.opt_index > rv.k]
    A["out_of_range"] = [dict(item=r.item_uid, model=r.model, temp=float(r.temp), trial=int(r.trial),
                              raw=str(r.raw_response), parsed=r.parsed_letter, k=int(kmap[r.item_uid]))
                         for r in oor.itertuples()]
    # duplicates in the complete runner log
    lg = pd.read_csv(RAW_LOG, encoding="utf-8-sig")
    A["log_rows"] = int(len(lg))
    lg["is_err"] = lg.raw_response.fillna("").astype(str).str.startswith("__ERROR__")
    A["log_error_rows"] = int(lg.is_err.sum())
    ok = lg[~lg.is_err]
    key = ["item_uid", "model", "temp", "trial"]
    dup = ok[ok.duplicated(key, keep=False)]
    A["dup_success_rows"] = int(len(dup)); A["dup_keys"] = int(dup[key].drop_duplicates().shape[0])
    A["dup_by_model"] = dup.drop_duplicates(key).groupby("model").size().astype(int).to_dict()
    disagree = dup.groupby(key).parsed_letter.nunique(dropna=False)
    A["dup_keys_with_differing_answers"] = int((disagree > 1).sum())
    # does keep-first change anything?
    first = ok.drop_duplicates(key, keep="first")
    last = ok.drop_duplicates(key, keep="last")
    A["n_keep_first"] = int(len(first)); A["n_keep_last"] = int(len(last))
    # category x stratum x year
    pr["year"] = pr.item_uid.str.split("_").str[1].str.split("-").str[0].astype(int)
    A["category_by_stratum"] = {f"{c}|{s}": int(n) for (c, s), n in pr.groupby(["category", "recent"]).size().items()}
    A["category_by_year"] = {f"{c}|{y}": int(n) for (c, y), n in pr.groupby(["category", "year"]).size().items()}
    A["k_by_category"] = {f"{c}|{k}": int(n) for (c, k), n in pr.groupby(["category", "k"]).size().items()}
    R["A_audit"] = A

    # ===================== B. residual items
    log("B. unanimous-but-wrong residual items")
    Bres = {}
    for m in MODELS:
        lst = []
        for it, g in cells[cells.model == m].groupby("item"):
            uw = g[(g.phat >= 0.999) & (g.modal_correct == 0)]
            if len(uw):
                rr = raw[(raw.item_uid == it) & (raw.model == m) & raw.valid]
                dist = rr.parsed_letter.value_counts().to_dict()
                lst.append(dict(item=it, category=g.category.iloc[0], key=g.key.iloc[0],
                                unanimous_wrong_temps=sorted(uw.temp.tolist()),
                                wrong_answers=sorted(set(uw.modal.tolist())),
                                pooled_dist={k: int(v) for k, v in dist.items()},
                                pooled_trials=int(len(rr))))
        Bres[m] = lst
    R["B_residual"] = Bres
    distinct = sorted({d["item"] for m in MODELS for d in Bres[m]})
    R["B_distinct_items"] = distinct
    # verification of the two worked examples in the Results text
    ex = {}
    for it_short in ["114_2025-AM-75", "110_2021"]:
        pass
    for it in distinct:
        ex[it] = {}
        for m in MODELS:
            rr = raw[(raw.item_uid == it) & (raw.model == m) & raw.valid]
            vc = rr.parsed_letter.value_counts()
            ex[it][m] = dict(modal=vc.idxmax(), frac=round(float(vc.max() / len(rr)), 4), n=int(len(rr)),
                             dist={k: int(v) for k, v in vc.items()})
    R["B_pooled_modal_by_item_model"] = ex

    # ===================== C. correlation
    log("C. consistency-accuracy correlation (unit of analysis / dependence)")
    C = {}
    it_all = cells.groupby("item").agg(acc=("accuracy", "mean"), nci=("NCI", "mean"))
    r, p = stats.spearmanr(it_all.acc, it_all.nci); C["item_pooled"] = dict(rho=float(r), p=float(p), n=len(it_all))
    r, p = stats.spearmanr(cells.accuracy, cells.NCI); C["cell_pooled"] = dict(rho=float(r), p=float(p), n=len(cells))
    for m in MODELS:
        cm = cells[cells.model == m]
        im = cm.groupby("item").agg(acc=("accuracy", "mean"), nci=("NCI", "mean"))
        r, p = stats.spearmanr(im.acc, im.nci)
        C[f"item_within:{m}"] = dict(rho=float(r), p=float(p), n=len(im),
                                     n_items_nci_lt1=int((im.nci < 1 - 1e-12).sum()),
                                     n_items_acc_lt1=int((im.acc < 1 - 1e-12).sum()))
        r, p = stats.spearmanr(cm.accuracy, cm.NCI)
        C[f"cell_within:{m}"] = dict(rho=float(r), p=float(p), n=len(cm))
        for t in TEMPS[m]:
            ct = cm[cm.temp == t]
            if ct.NCI.nunique() < 2 or ct.accuracy.nunique() < 2:
                C[f"stratum:{m}:{t}"] = dict(rho=None, p=None, n=len(ct), note="constant input (ceiling)",
                                             n_nci_lt1=int((ct.NCI < 1 - 1e-12).sum()))
            else:
                r, p = stats.spearmanr(ct.accuracy, ct.NCI)
                C[f"stratum:{m}:{t}"] = dict(rho=float(r), p=float(p), n=len(ct),
                                             n_nci_lt1=int((ct.NCI < 1 - 1e-12).sum()))
    kinds = ["pooled|item", "pooled|cell"] + [f"within:{m}|item" for m in MODELS] + [f"within:{m}|cell" for m in MODELS]
    nch = 8; per = B // nch
    tasks = [(k, ch, per, SEED + 1000 * i + ch) for i, k in enumerate(kinds) for ch in range(nch)]
    out = run_parallel(_boot_corr_chunk, tasks, "C. item-cluster bootstrap of Spearman rho")
    for k in kinds:
        vals = sum((out[(k, ch)] for ch in range(nch)), [])
        lo, hi, nv = pct_ci(vals); C[f"boot:{k}"] = dict(lo=lo, hi=hi, n_valid=nv)
    R["C_correlation"] = C

    # ===================== D. calibration
    log("D. calibration: ECE bins + Brier")
    D = {}
    edges = np.linspace(0, 1, 11)
    for m in MODELS:
        cm = cells[cells.model == m]; conf = cm.phat.values; y = cm.modal_correct.values; N = len(cm)
        bins = []; ece = 0.0; rel = 0.0; res = 0.0; ybar = y.mean()
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            msk = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            nk = int(msk.sum())
            if nk:
                ck, ok_ = conf[msk].mean(), y[msk].mean()
                ece += nk / N * abs(ok_ - ck); rel += nk * (ck - ok_) ** 2; res += nk * (ok_ - ybar) ** 2
                bins.append(dict(bin=f"({lo:.1f},{hi:.1f}]" if i else f"[{lo:.1f},{hi:.1f}]", n=nk,
                                 mean_conf=float(ck), acc=float(ok_)))
            else:
                bins.append(dict(bin=f"({lo:.1f},{hi:.1f}]" if i else f"[{lo:.1f},{hi:.1f}]", n=0, mean_conf=None, acc=None))
        brier = float(np.mean((conf - y) ** 2))
        n_top = int((conf >= 0.999).sum())
        D[m] = dict(N=N, ECE=float(ece), bins=bins, Brier=brier, reliability=rel / N, resolution=res / N,
                    uncertainty=float(ybar * (1 - ybar)), base_acc=float(ybar),
                    n_conf_eq1=n_top, acc_when_conf_eq1=float(y[conf >= 0.999].mean()),
                    brier_ref_constant=float(ybar * (1 - ybar)))
    R["D_calibration"] = D

    # ===================== E. triage
    log("E. triage AUC: legacy vs tie-corrected")
    E = {"per_temp": {}, "summary": {}}
    for m in MODELS:
        cm = cells[cells.model == m]
        leg, exp_, best, worst, roc, nerr, ntie = [], [], [], [], [], [], []
        for t in TEMPS[m]:
            ct = cm[cm.temp == t]
            err = (ct.modal_correct.values == 0).astype(int)
            a_leg = legacy_auc(ct)
            a_exp = capture_auc(ct.NCI.values, err, "expected")
            a_b = capture_auc(ct.NCI.values, err, "best"); a_w = capture_auc(ct.NCI.values, err, "worst")
            a_roc = roc_auc_ties_half(ct.NCI.values, err)
            n1 = int((np.round(ct.NCI.values, 12) >= 1 - 1e-12).sum())
            e1 = int(err[np.round(ct.NCI.values, 12) >= 1 - 1e-12].sum())
            E["per_temp"][f"{m}|{t}"] = dict(n_items=len(ct), n_errors=int(err.sum()), n_nci_eq1=n1,
                                             n_errors_at_nci_eq1=e1, legacy=a_leg, expected=a_exp,
                                             best=a_b, worst=a_w, roc_ties_half=a_roc)
            for L, v in [(leg, a_leg), (exp_, a_exp), (best, a_b), (worst, a_w), (roc, a_roc)]:
                if v is not None: L.append(v)
        E["summary"][m] = dict(legacy_mean=float(np.mean(leg)), expected_mean=float(np.mean(exp_)),
                               best_mean=float(np.mean(best)), worst_mean=float(np.mean(worst)),
                               roc_mean=float(np.mean(roc)) if roc else None)
        # sensitivity: exclude temperatures at which every item is tied (no ranking information at all)
        informative = [E["per_temp"][f"{m}|{t}"]["expected"] for t in TEMPS[m]
                       if E["per_temp"][f"{m}|{t}"]["n_nci_eq1"] < E["per_temp"][f"{m}|{t}"]["n_items"]
                       and E["per_temp"][f"{m}|{t}"]["expected"] is not None]
        E["summary"][m]["expected_mean_informative_temps"] = float(np.mean(informative)) if informative else None
        E["summary"][m]["n_informative_temps"] = len(informative)

    nch = 8; per = B // nch
    tasks = [(m, ch, per, SEED + 50000 + 100 * i + ch) for i, m in enumerate(MODELS) for ch in range(nch)]
    out = run_parallel(_boot_triage_chunk, tasks, "E. bootstrap (joint item resampling, tie-corrected)")
    for m in MODELS:
        reps = sum((out[(m, ch)] for ch in range(nch)), [])
        means = [r["mean"] for r in reps]
        lo, hi, nv = pct_ci(means)
        skipped = [r["n_temps_skipped"] for r in reps]
        E["summary"][m].update(boot_lo=lo, boot_hi=hi, boot_valid=nv, boot_dropped=B - nv,
                               boot_reps_with_any_temp_skipped=int(sum(s > 0 for s in skipped)),
                               boot_temp_skips_total=int(sum(skipped)))
        # complete-case sensitivity: only replicates in which EVERY temperature has >= 1 error
        cc = [r["mean"] for r in reps if r["n_temps_skipped"] == 0]
        if len(cc) > 50:
            lo2, hi2, n2 = pct_ci(cc); E["summary"][m].update(cc_lo=lo2, cc_hi=hi2, cc_n=n2)
        else:
            E["summary"][m].update(cc_lo=None, cc_hi=None, cc_n=len(cc))
        # per-temperature bootstrap CIs
        for t in TEMPS[m]:
            vals = [r["per_t"][str(t)] for r in reps]
            lo3, hi3, n3 = pct_ci(vals)
            E["per_temp"][f"{m}|{t}"].update(boot_lo=lo3, boot_hi=hi3, boot_valid=n3)
    tasks = [(m, ch, per, 20260703 + 100 * i + ch) for i, m in enumerate(MODELS) for ch in range(nch)]
    out = run_parallel(_boot_legacy_chunk, tasks, "E'. legacy-procedure bootstrap (replication attempt)")
    for m in MODELS:
        vals = sum((out[(m, ch)] for ch in range(nch)), [])
        lo, hi, nv = pct_ci(vals)
        E["summary"][m].update(legacy_boot_lo=lo, legacy_boot_hi=hi, legacy_boot_median=float(np.median([v for v in vals if v is not None])))
    R["E_triage"] = E

    (OUT / "revision_results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float))
    log(f"written {OUT/'revision_results.json'}")

if __name__ == "__main__":
    main()
