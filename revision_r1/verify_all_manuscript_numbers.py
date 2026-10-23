#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute every remaining number in the submitted manuscript (Tables 2-3, Friedman, Kruskal-Wallis,
high-confidence accuracy, contamination strata) from raw data, to confirm or correct before revision."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import revision_analyses as ra

cells = pd.read_pickle(ra.OUT / "cells.pkl")
pr = pd.read_csv(ra.PROMPTS, encoding="utf-8-sig")
rec = dict(zip(pr.item_uid, pr.recent))
cells["recent"] = cells["item"].map(rec)

print("== Table 2: consistency mean (SD, population) and accuracy by model x temperature ==")
for m in ra.MODELS:
    for t in ra.TEMPS[m]:
        c = cells[(cells.model == m) & (cells.temp == t)]
        print(f"  {ra.LAB[m]:7s} T={t}: NCI {c.NCI.mean():.3f} (SD {c.NCI.std(ddof=0):.3f}; sample SD {c.NCI.std(ddof=1):.3f})  acc {c.accuracy.mean():.3f}  n={len(c)}")

print("\n== Friedman (items as blocks) ==")
for m in ra.MODELS:
    w = cells[cells.model == m].pivot(index="item", columns="temp", values="NCI")
    for label, cols in [("full", ra.TEMPS[m]), ("common", [0.0, 0.5, 1.0])]:
        x = w[cols].values
        chi, p = stats.friedmanchisquare(*[x[:, j] for j in range(x.shape[1])])
        n, k = x.shape; W = chi / (n * (k - 1))
        print(f"  {ra.LAB[m]:7s} {label:6s}: chi2({k-1}) = {chi:.2f}, p = {p:.2g}, W = {W:.2f}")

print("\n== Kruskal-Wallis by category ==")
H, p = stats.kruskal(*[g.NCI.values for _, g in cells.groupby("category")])
print(f"  cell level (990 cells): H = {H:.2f}, p = {p:.3f}")
it = cells.groupby(["item", "category"]).NCI.mean().reset_index()
H2, p2 = stats.kruskal(*[g.NCI.values for _, g in it.groupby("category")])
print(f"  item level (90 items, pooled): H = {H2:.2f}, p = {p2:.3f}")

print("\n== Table 3: by model x category (temperature-pooled cells) ==")
for m in ra.MODELS:
    for cat in ["Knowledge", "Priority", "EthicalLegal"]:
        c = cells[(cells.model == m) & (cells.category == cat)]
        print(f"  {ra.LAB[m]:7s} {cat:12s} NCI {c.NCI.mean():.3f}  acc {c.accuracy.mean():.3f}  n_cells={len(c)}")

print("\n== Accuracy of modal answer when modal frequency >= 0.8 ==")
for m in ra.MODELS:
    c = cells[(cells.model == m) & (cells.phat >= 0.8)]
    print(f"  {ra.LAB[m]:7s}: {c.modal_correct.mean():.3f} (n={len(c)})")

print("\n== Contamination strata (temperature-pooled) ==")
for m in ra.MODELS:
    for s in ["Y", "N"]:
        c = cells[(cells.model == m) & (cells.recent == s)]
        print(f"  {ra.LAB[m]:7s} recent={s}: NCI {c.NCI.mean():.3f} acc {c.accuracy.mean():.3f} items={c['item'].nunique()}")

print("\n== pooled accuracy by model (all temps) ==")
for m in ra.MODELS:
    print(f"  {ra.LAB[m]:7s}: {cells[cells.model == m].accuracy.mean():.4f}; acc range over temps "
          f"{min(cells[(cells.model==m)&(cells.temp==t)].accuracy.mean() for t in ra.TEMPS[m]):.3f}-"
          f"{max(cells[(cells.model==m)&(cells.temp==t)].accuracy.mean() for t in ra.TEMPS[m]):.3f}")

print("\n== items unanimous at T=0 ==")
for m in ra.MODELS:
    c = cells[(cells.model == m) & (cells.temp == 0.0)]
    print(f"  {ra.LAB[m]:7s}: {(c.phat >= 0.999).sum()}/90 unanimous; errors {int((c.modal_correct==0).sum())}")
