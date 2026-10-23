#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitivity analyses restricted to items whose prompt text was information-complete
(post hoc extraction audit; classification adjudicated item by item from the prompts actually sent)."""
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import revision_analyses as ra

OUT = Path(os.environ.get("NCI_OUT", "outputs")).resolve()
CONTENT = ["107_2018-AM-78","108_2019-AM-45","108_2019-PM-49","108_2019-PM-66","108_2019-PM-70",
           "109_2020-AM-112","109_2020-AM-64","109_2020-PM-60","109_2020-PM-69","110_2021-AM-65",
           "110_2021-PM-71","111_2022-AM-60","111_2022-AM-73","111_2022-AM-92","111_2022-PM-48",
           "111_2022-PM-70","112_2023-AM-69","112_2023-AM-95","112_2023-PM-84","113_2024-PM-60"]
CONTEXT = ["109_2020-AM-112","111_2022-AM-118","111_2022-AM-92","111_2022-PM-97","112_2023-AM-95"]
COSMETIC = ["107_2018-AM-60","108_2019-AM-63","108_2019-PM-41","109_2020-PM-35","110_2021-PM-40",
            "110_2021-PM-43","111_2022-AM-75","112_2023-PM-45","113_2024-AM-53","113_2024-AM-58",
            "114_2025-PM-29","114_2025-PM-56","115_2026-AM-11","115_2026-AM-15","115_2026-AM-20",
            "115_2026-AM-28","115_2026-AM-29","115_2026-AM-31","115_2026-AM-36","115_2026-AM-37",
            "115_2026-AM-47","115_2026-AM-51","115_2026-AM-53","115_2026-AM-62","115_2026-PM-69"]
DESC = {
 "107_2018-AM-78": "numerals dropped (child's age; family size)",
 "108_2019-AM-45": "numerals dropped in three options (time since onset; days; dose multiple)",
 "108_2019-PM-49": "numerals dropped (CKD stage; alcohol frequency and amount)",
 "108_2019-PM-66": "numerals dropped (living alone; number of nightly toilet trips)",
 "108_2019-PM-70": "numeral dropped in a distractor (infectious-disease class)",
 "109_2020-AM-112": "case vignette absent (situational-set sub-question); numeral dropped in an option",
 "109_2020-AM-64": "numeral dropped in a distractor (timing of first dispatch)",
 "109_2020-PM-60": "numeral dropped (child's age)",
 "109_2020-PM-69": "numeral dropped in a distractor (number of designated psychiatrists)",
 "110_2021-AM-65": "numerals dropped (household size; months since diagnosis)",
 "110_2021-PM-71": "numeral dropped in a distractor (number of triage assessments)",
 "111_2022-AM-60": "numerals dropped (school grade; oxygen flow; weeks to ceremony)",
 "111_2022-AM-73": "one character dropped in an option ('[pain] control')",
 "111_2022-AM-92": "case vignette absent; numeral dropped (postoperative day)",
 "111_2022-PM-48": "key word garbled in the stem ('lower limb')",
 "111_2022-PM-70": "numeral dropped (living alone)",
 "112_2023-AM-69": "numeral dropped (living alone)",
 "112_2023-AM-95": "case vignette absent; numeral dropped (week of admission)",
 "112_2023-PM-84": "numerals dropped (months since admission; 'can do [alone]')",
 "113_2024-PM-60": "numeral dropped (gestational day)",
 "111_2022-AM-118": "case vignette absent (situational-set sub-question)",
 "111_2022-PM-97": "case vignette absent (situational-set sub-question)",
}

def main():
    cells = pd.read_pickle(OUT / "cells.pkl")
    pr = pd.read_csv(ra.PROMPTS, encoding="utf-8-sig")
    allitems = sorted(pr.item_uid)
    incomplete = sorted(set(CONTENT) | set(CONTEXT))
    assert len(set(CONTENT) | set(CONTEXT) | set(COSMETIC)) == len(incomplete) + len(COSMETIC)
    assert all(i in allitems for i in incomplete + COSMETIC)
    status = {i: ("incomplete" if i in incomplete else "cosmetic" if i in COSMETIC else "clean") for i in allitems}
    pd.DataFrame([dict(item=i, status=status[i], content_artifact=i in CONTENT, context_missing=i in CONTEXT,
                       description=DESC.get(i, ""), year=int(i.split("_")[1].split("-")[0]),
                       category=pr.set_index("item_uid").loc[i, "category"]) for i in allitems]
                 ).to_csv(OUT / "extraction_audit_classification.csv", index=False)
    print({k: sum(v == k for v in status.values()) for k in ["clean", "cosmetic", "incomplete"]})
    yr = pd.DataFrame([dict(year=int(i.split("_")[1].split("-")[0]), st=status[i]) for i in allitems])
    print(pd.crosstab(yr.year, yr.st))

    subsets = {"all_90": allitems,
               "complete_68": [i for i in allitems if status[i] != "incomplete"],
               "clean_43": [i for i in allitems if status[i] == "clean"]}
    S = {}
    rng = np.random.default_rng(ra.SEED + 7)
    for name, items in subsets.items():
        c = cells[cells["item"].isin(items)]
        res = dict(n_items=len(items))
        for m in ra.MODELS:
            cm = c[c.model == m]
            # residual (unanimous at any temperature & wrong)
            resid = sorted(it for it, g in cm.groupby("item") if ((g.phat >= 0.999) & (g.modal_correct == 0)).any())
            conf, y = cm.phat.values, cm.modal_correct.values
            edges = np.linspace(0, 1, 11); ece = 0.0
            for k in range(10):
                lo, hi = edges[k], edges[k + 1]
                msk = (conf > lo) & (conf <= hi) if k else (conf >= lo) & (conf <= hi)
                if msk.sum(): ece += msk.sum() / len(conf) * abs(y[msk].mean() - conf[msk].mean())
            aucs = []
            for t in ra.TEMPS[m]:
                ct = cm[cm.temp == t]
                a = ra.capture_auc(ct.NCI.values, (ct.modal_correct.values == 0).astype(int), "expected")
                if a is not None: aucs.append(a)
            im = cm.groupby("item").agg(acc=("accuracy", "mean"), nci=("NCI", "mean"))
            rho = stats.spearmanr(im.acc, im.nci).statistic if im.nci.nunique() > 1 else None
            # bootstrap CI for the temperature-averaged tie-corrected AUC (joint item resampling)
            its = np.array(sorted(cm["item"].unique()))
            by_t = {t: cm[cm.temp == t].set_index("item").loc[its] for t in ra.TEMPS[m]}
            boots = []
            for _ in range(ra.B):
                idx = rng.integers(0, len(its), len(its)); v = []
                for t in ra.TEMPS[m]:
                    d = by_t[t].iloc[idx]
                    a = ra.capture_auc(d.NCI.values, (d.modal_correct.values == 0).astype(int), "expected")
                    if a is not None: v.append(a)
                if v: boots.append(np.mean(v))
            res[m] = dict(accuracy=float(cm.accuracy.mean()), residual_n=len(resid), residual_items=resid,
                          ECE=float(ece), Brier=float(np.mean((conf - y) ** 2)),
                          AUC_tie_corrected=float(np.mean(aucs)) if aucs else None,
                          AUC_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
                          AUC_boot_valid=len(boots),
                          rho_within_item=None if rho is None or np.isnan(rho) else float(rho))
        ip = c.groupby("item").agg(acc=("accuracy", "mean"), nci=("NCI", "mean"))
        res["rho_item_pooled"] = float(stats.spearmanr(ip.acc, ip.nci).statistic)
        S[name] = res
        print(f"\n== {name} (n={len(items)}) ==  pooled item rho={res['rho_item_pooled']:.3f}")
        for m in ra.MODELS:
            r = res[m]
            fa = lambda x: "NA" if x is None else f"{x:.3f}"
            ci = "NA" if r['AUC_ci'] is None else [round(x, 2) for x in r['AUC_ci']]
            print(f"  {ra.LAB[m]:7s} acc={r['accuracy']:.3f} resid={r['residual_n']} {r['residual_items']} ECE={r['ECE']:.3f} "
                  f"Brier={r['Brier']:.4f} AUC={fa(r['AUC_tie_corrected'])} CI={ci} boot_valid={r['AUC_boot_valid']} "
                  f"rho_within={fa(r['rho_within_item'])}")
    (OUT / "sensitivity_subsets.json").write_text(json.dumps(S, indent=1))

if __name__ == "__main__":
    main()
