"""
NCI analysis script - reproduces all statistics reported in the manuscript.

This script reads cleaned raw responses and reproduces:
  - Per-cell Nursing Consistency Index (NCI)
  - Friedman tests for temperature effect within each model
  - Kruskal-Wallis test on scenario-level mean NCI for category effect
  - Fleiss' kappa for convergent validity (uses all 9,836 valid responses)
  - ICC(2,1) on a balanced 30 x 22 sub-sample (cross-reference only)
  - NCI vs accuracy correlations

Usage:
    python scripts/analyze.py

Inputs (from data/):
    raw_responses.csv
    scenarios.csv

Outputs:
    Console output of all reported statistics.

Requires: numpy, pandas, scipy
"""

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "raw_responses.csv"

VALID = {"A", "B", "C", "D"}
N_OPTIONS = 4


def calc_nci(answers):
    """Nursing Consistency Index = 1 - H(R)/log2(k)."""
    n = len(answers)
    if n == 0:
        return float("nan")
    counts = Counter(answers)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
    return 1.0 - h / math.log2(N_OPTIONS)


def fleiss_kappa(per_subject_assignments, n_categories=N_OPTIONS):
    """Fleiss' kappa for nominal data, allowing variable raters per subject."""
    p_i = []
    grand_n = 0
    total_counts = {}
    for assignment in per_subject_assignments:
        n_i = sum(assignment.values())
        if n_i < 2:
            continue
        total = sum(c * (c - 1) for c in assignment.values())
        p_i.append(total / (n_i * (n_i - 1)))
        for cat, c in assignment.items():
            total_counts[cat] = total_counts.get(cat, 0) + c
            grand_n += c
    if not p_i or grand_n == 0:
        return float("nan")
    P_bar = float(np.mean(p_i))
    p_j = {cat: total_counts.get(cat, 0) / grand_n for cat in ["A", "B", "C", "D"]}
    P_e = sum(p**2 for p in p_j.values())
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def icc_2_1(matrix):
    """ICC(2,1) absolute agreement, two-way random-effects, single-measure."""
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape  # n subjects, k raters
    grand_mean = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    SST = ((matrix - grand_mean) ** 2).sum()
    SSB = k * ((row_means - grand_mean) ** 2).sum()
    SSC = n * ((col_means - grand_mean) ** 2).sum()
    SSE = SST - SSB - SSC
    MSR = SSB / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)


def main():
    if not RAW_PATH.exists():
        print(f"ERROR: {RAW_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(RAW_PATH)
    df_valid = df[df["is_valid"]].copy()

    # ── Per-cell NCI ─────────────────────────────────
    cells = []
    for (qid, cat, model, temp), g in df_valid.groupby(
        ["question_id", "category", "model", "temperature"]
    ):
        cells.append(
            {
                "question_id": qid,
                "category": cat,
                "model": model,
                "temperature": temp,
                "NCI": calc_nci(g["response"].tolist()),
                "accuracy": (g["response"] == g["intended_answer"].iloc[0]).mean(),
            }
        )
    nci_df = pd.DataFrame(cells)

    # ── Cell-level descriptives ──────────────────────
    print("=" * 70)
    print("Mean (SD) NCI by model x temperature")
    print("=" * 70)
    desc = (
        nci_df.groupby(["model", "temperature"])["NCI"]
        .agg(["mean", "std"])
        .round(4)
    )
    print(desc.to_string())

    # ── Friedman test for temperature effect ─────────
    print("\n" + "=" * 70)
    print("Friedman test: temperature effect within each model")
    print("=" * 70)
    for model in ["gpt", "claude", "gemini"]:
        sub = nci_df[nci_df["model"] == model]
        temps = sorted(sub["temperature"].unique())
        groups = [sub[sub["temperature"] == t].sort_values("question_id")["NCI"].values
                  for t in temps]
        chi2, p = stats.friedmanchisquare(*groups)
        df_v = len(groups) - 1
        print(f"  {model:8s}: chi2({df_v}) = {chi2:.3f}, p = {p:.4f} (k={len(groups)} temps)")

    # ── Kruskal-Wallis on scenario-level means ──────
    print("\n" + "=" * 70)
    print("Kruskal-Wallis: judgment category effect (scenario-level means)")
    print("=" * 70)
    sc_mean = nci_df.groupby(["question_id", "category"])["NCI"].mean().reset_index()
    g_kw = [sc_mean[sc_mean["category"] == c]["NCI"].values
            for c in ["Knowledge", "Ethical", "Priority"]]
    h, p = stats.kruskal(*g_kw)
    print(f"  H(2) = {h:.4f}, p = {p:.4f}  (n=10 per category)")
    for c in ["Knowledge", "Ethical", "Priority"]:
        m = sc_mean[sc_mean["category"] == c]["NCI"].mean()
        print(f"    {c:12s}: mean NCI = {m:.4f}")

    # ── Fleiss' kappa (using ALL valid data) ────────
    print("\n" + "=" * 70)
    print("Fleiss' kappa per (model, temperature) cell — all 9,836 valid responses")
    print("=" * 70)
    kappa_rows = []
    for (model, temp), g in df_valid.groupby(["model", "temperature"]):
        per_sc = []
        for qid, gg in g.groupby("question_id"):
            counts = Counter(gg["response"].tolist())
            per_sc.append({c: counts.get(c, 0) for c in ["A", "B", "C", "D"]})
        kappa = fleiss_kappa(per_sc)
        mean_nci = nci_df[(nci_df["model"] == model) & (nci_df["temperature"] == temp)]["NCI"].mean()
        kappa_rows.append({"model": model, "temperature": temp,
                          "fleiss_kappa": kappa, "mean_NCI": mean_nci})
        print(f"  {model:8s} T={temp}: kappa = {kappa:.4f}")
    kappa_df = pd.DataFrame(kappa_rows)
    r_p, p_p = stats.pearsonr(kappa_df["mean_NCI"], kappa_df["fleiss_kappa"])
    r_s, p_s = stats.spearmanr(kappa_df["mean_NCI"], kappa_df["fleiss_kappa"])
    print(f"\n  NCI vs Fleiss' kappa: Pearson r = {r_p:.4f} (p={p_p:.4g}), "
          f"Spearman rho = {r_s:.4f} (p={p_s:.4g}), n={len(kappa_df)} cells")

    # ── ICC(2,1) on balanced 30x22 sub-sample (cross-reference) ──
    print("\n" + "=" * 70)
    print("ICC(2,1) on balanced 30 x 22 sub-sample (cross-reference only)")
    print("=" * 70)
    icc_rows = []
    enc = {"A": 1, "B": 2, "C": 3, "D": 4}
    for (model, temp), g in df_valid.groupby(["model", "temperature"]):
        mat = []
        for qid, gg in g.groupby("question_id"):
            answers = [enc[a] for a in gg["response"].tolist()][:22]
            if len(answers) < 22:
                continue
            mat.append(answers)
        if len(mat) < 30:
            continue
        mat = np.array(mat[:30])
        icc = icc_2_1(mat)
        mean_nci = nci_df[(nci_df["model"] == model) & (nci_df["temperature"] == temp)]["NCI"].mean()
        icc_rows.append({"model": model, "temperature": temp, "ICC_2_1": icc, "mean_NCI": mean_nci})
        print(f"  {model:8s} T={temp}: ICC(2,1) = {icc:.4f}")
    icc_df = pd.DataFrame(icc_rows)
    r_p, p_p = stats.pearsonr(icc_df["mean_NCI"], icc_df["ICC_2_1"])
    r_s, p_s = stats.spearmanr(icc_df["mean_NCI"], icc_df["ICC_2_1"])
    print(f"\n  NCI vs ICC(2,1): Pearson r = {r_p:.4f} (p={p_p:.4g}), "
          f"Spearman rho = {r_s:.4f} (p={p_s:.4g}), n={len(icc_df)} cells")

    # ── NCI vs Accuracy by model ─────────────────────
    print("\n" + "=" * 70)
    print("NCI vs Accuracy correlation per model")
    print("=" * 70)
    for model in ["gpt", "claude", "gemini"]:
        sub = nci_df[nci_df["model"] == model]
        r, p = stats.pearsonr(sub["NCI"], sub["accuracy"])
        print(f"  {model:8s}: Pearson r = {r:.4f}, p = {p:.4g}, n = {len(sub)} cells")


if __name__ == "__main__":
    main()
