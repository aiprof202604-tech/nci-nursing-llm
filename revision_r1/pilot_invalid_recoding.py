#!/usr/bin/env python3
"""English pilot (Supplementary S1.3): sensitivity of mean consistency to the 64 excluded responses.
Primary convention: consistency is computed over valid responses only (A-D), k = 4.
Conservative recoding: the 64 excluded responses (62 API rate-limit errors, 2 non-conforming outputs) are retained as
one additional answer category, so each cell has 30 responses over k = 5 categories (four options + "invalid").
Requests rejected as out of range (Claude x T = 1.5, never observed) are not part of the 9,900 in-range calls and
are omitted. Mean = unweighted mean over the model's observed (scenario, temperature) cells.
Input: data/raw_responses.csv of the repository. No API calls. Environment: NCI_REPO (default ./nci-nursing-llm)."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(os.environ.get("NCI_REPO", "nci-nursing-llm")).resolve()
VALID = set("ABCD")


def consistency(responses, k):
    p = pd.Series(responses).value_counts(normalize=True).to_numpy()
    return 1.0 - float(-(p * np.log2(p)).sum()) / np.log2(k)


def main():
    d = pd.read_csv(REPO / "data" / "raw_responses.csv")
    d = d[d.response != "API_REJECTED_T_OUT_OF_RANGE"]
    print(f"in-range calls: {len(d)}; excluded responses: {int((~d.response.isin(VALID)).sum())}")
    for m in ["claude", "gpt", "gemini"]:
        valid_only, recoded = [], []
        for _, g in d[d.model == m].groupby(["question_id", "temperature"]):
            r = g.response.tolist()
            v = [x for x in r if x in VALID]
            if v:
                valid_only.append(consistency(v, 4))
            recoded.append(consistency([x if x in VALID else "INVALID" for x in r], 5))
        print(f"{m:7s} cells = {len(recoded):3d}  valid-only (k = 4): {np.mean(valid_only):.3f}  "
              f"invalid as a category (k = 5): {np.mean(recoded):.3f}")


if __name__ == "__main__":
    main()
