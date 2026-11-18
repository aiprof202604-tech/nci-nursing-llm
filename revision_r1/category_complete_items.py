#!/usr/bin/env python3
"""Supplementary S14 / Table S12 (second revision): judgment-category comparison restricted to the 68
information-complete items. Mean consistency and mean accuracy by model x category over temperature-pooled cells
(as in Table 3), and the item-level Kruskal-Wallis test on item mean consistency, for all 90 and for the 68 items.
Exact fractions are rounded half-up. Run revision_analyses.py first (it writes outputs/cells_revision.csv).
Inputs: NCI_OUT/cells_revision.csv, extraction_audit_classification.csv (this folder), and the public raw data
(NCI_REPO). No API calls. Output: NCI_OUT/category_complete_items.json."""
import json, os
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
import pandas as pd
from scipy import stats

REPO = Path(os.environ.get("NCI_REPO", "nci-nursing-llm")).resolve()
OUT = Path(os.environ.get("NCI_OUT", "outputs")).resolve()
HERE = Path(__file__).resolve().parent
RAW = REPO / "study2_japanese_national_exam" / "exp5_results" / "raw_dedup_29700.csv"


def half_up(fr, nd=3):
    return str((Decimal(fr.numerator) / Decimal(fr.denominator)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def main():
    cells = pd.read_csv(OUT / "cells_revision.csv")
    aud = pd.read_csv(HERE / "extraction_audit_classification.csv")
    raw = pd.read_csv(RAW, encoding="utf-8-sig")
    raw = raw[raw.parsed_letter.notna() & (raw.parsed_letter.astype(str).str.strip() != "")]
    complete = set(aud[aud.status != "incomplete"].item)
    res = {}
    for k, (lab, items) in enumerate([("all_90", set(cells.item)), ("complete_68", complete)], 1):
        c = cells[cells.item.isin(items)]
        r = raw[raw.item_uid.isin(items)]
        acc = {key: Fraction(int((g.parsed_letter == g.key_letter).sum()), len(g))
               for key, g in r.groupby(["item_uid", "model", "temp"])}
        tab = {}
        for (m, cat), g in c.groupby(["model", "category"]):
            a = [acc[(row.item, row.model, row.temp)] for row in g.itertuples()]
            tab[f"{m}|{cat}"] = dict(n_items=int(g.item.nunique()), consistency=round(float(g.NCI.mean()), 3),
                                     accuracy=half_up(sum(a, Fraction(0)) / len(a)))
        it = c.groupby(["item", "category"]).NCI.mean().reset_index()
        H, p = stats.kruskal(*[gg.NCI.values for _, gg in it.groupby("category")])
        res[lab] = dict(n_items_by_category={kk: int(v) for kk, v in it.groupby("category").size().items()},
                        table=tab, kruskal_H=round(float(H), 2), kruskal_p=round(float(p), 2))
        print(f"[{k}/2] {lab}: {res[lab]['n_items_by_category']}  H = {res[lab]['kruskal_H']}, p = {res[lab]['kruskal_p']}",
              flush=True)
        for kk, v in tab.items():
            print("     ", kk, v)
    (OUT / "category_complete_items.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
