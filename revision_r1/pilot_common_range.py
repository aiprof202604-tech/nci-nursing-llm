#!/usr/bin/env python3
"""English pilot (Supplementary S1.4): Friedman tests of the temperature effect on the common range
T in {0.0, 0.5, 1.0}, scenarios as blocks, from the repository's per-cell file data/nci_summary.csv.
All other pilot statistics are printed by scripts/analyze.py of the repository. No API calls.
Environment: NCI_REPO (default ./nci-nursing-llm)."""
import os
from pathlib import Path
import pandas as pd
from scipy import stats

REPO = Path(os.environ.get("NCI_REPO", "nci-nursing-llm")).resolve()

def main():
    d = pd.read_csv(REPO / "data" / "nci_summary.csv")
    for m in sorted(d.model.unique()):
        sub = d[(d.model == m) & (d.temperature.isin([0.0, 0.5, 1.0]))].pivot(index="question_id", columns="temperature",
                                                                              values="NCI").dropna()
        r = stats.friedmanchisquare(*[sub[c] for c in sub.columns])
        print(f"{m:7s} n = {len(sub)} scenarios: chi2(2) = {r.statistic:.2f}, p = {r.pvalue:.4f}")

if __name__ == "__main__":
    main()
