# Reproducing the Study-2 statistics reported in the manuscript

This folder (`reanalysis_v31/`) contains the canonical reanalysis code and its
output, so every headline number in the manuscript can be reproduced from the
public raw responses already in this repository.

## One-command verification (for reviewers)
From inside this folder:
```
python3 verify_headline_numbers.py
```
It locates the raw responses automatically at
`../study2_japanese_national_exam/exp5_results/raw_dedup_29700.csv` and prints,
for each model, ECE, the unanimous-but-wrong residual, triage AUC, and
Spearman's rho, each checked against the value reported in the paper. Expected
final line: `ALL HEADLINE NUMBERS REPRODUCED`.

## Full reanalysis
```
python3 analysis_v31.py
```
Reads the raw responses for both studies (paths resolve to the repository root
automatically) and writes per-cell metrics, calibration/moderator/triage
outputs, and figures to `reanalysis_v31/outputs/`. `make_study2_figures.py`
renders the Study-2 manuscript figures from `cells_all.csv`.

## Files
- `analysis_v31.py` — three-layer reanalysis (calibration, moderators, triage)
  referenced in the Data availability statement. Produces per-cell metrics.
- `make_study2_figures.py` — renders Figures 1–3 from `cells_all.csv` (Study-2).
- `cells_all.csv` — per-cell metrics (NCI, accuracy, modal-answer frequency,
  majority-vote correctness) for every (study, model, temperature, item) cell;
  the direct input to every reported statistic.
- `verify_headline_numbers.py` — self-contained checker (see above).

## Handling of invalid responses (important)
The reported statistics use the convention **valid = a non-empty parsed
response**. Of the 29,700 Study-2 calls, 11 (0.04%) returned an empty response
and are excluded from the answer distributions; two further responses were
out-of-range characters and were retained as recorded. Because these thirteen
responses are a negligible fraction, all findings are robust to their handling.
Implemented in `analysis_v31.py` (`valid = parsed_letter.notna()`) and in
`verify_headline_numbers.py`.

## Relationship to `exp5_analyze.py`
`../study2_japanese_national_exam/exp5_analyze.py` is an earlier, exploratory
per-cell pass that treats an empty response as a distinct category (∅) with a
denominator of 30 rather than excluding it. Its summary tables therefore differ
by ≤0.004 for the two Knowledge cells that contain empty responses. It is
retained for transparency but is **not** the basis of the reported statistics;
the canonical pipeline is `analysis_v31.py` → `cells_all.csv` →
`make_study2_figures.py`.

## Calibration (ECE) bin convention
Ten equal-width bins on [0, 1] with right-closed intervals `(lo, hi]` (the first
bin includes 0). Confidence = modal-answer frequency; accuracy = majority-vote
correctness.
