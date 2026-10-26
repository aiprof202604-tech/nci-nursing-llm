# Supplementary code — revision analyses (ACI-2026-07-RA-0313, Round 1)

These scripts reproduce every statistic added or changed in the revised manuscript from the public raw data.
They make **no API calls**.

## Requirements
Python ≥ 3.12 with numpy, pandas, scipy, matplotlib, pillow (versions used: Python 3.13, numpy 2.5, pandas 3.0,
scipy 1.18, matplotlib 3.11). `verify_item_selection.py` additionally needs pdfplumber and pymupdf and network access
to www.mhlw.go.jp.

## Data
Inside this repository, run the scripts from the repository root with `export NCI_REPO=.` (for example,
`python3 revision_r1/revision_analyses.py`). As a stand-alone archive (Supplementary_Code_R1.zip), clone the study
repository next to this folder (or point `NCI_REPO` to it):

    git clone https://github.com/aiprof202604-tech/nci-nursing-llm
    export NCI_REPO=nci-nursing-llm      # default
    export NCI_OUT=outputs               # default

Inputs used: `study2_japanese_national_exam/exp5_results/raw_dedup_29700.csv`,
`study2_japanese_national_exam/exp5_results/raw.csv` (request log), and
`study2_japanese_national_exam/exp5_pool/exp5_prompts.csv`.

## Run (in this order)

    python3 revision_analyses.py                  # audit, residual items, correlations, calibration, triage (+ bootstrap)
    python3 sensitivity_subsets.py                # analyses on the 68 information-complete and 43 defect-free items
    python3 verify_all_manuscript_numbers.py      # other response-derived numbers: Tables 2-3, tests, year strata
    python3 consistently_wrong_by_temperature.py  # consistently wrong items excluding T = 0; temperatures per item
    python3 modal_tie_check.py                    # the single tied modal answer and its sensitivity (Supplementary S10.5)
    python3 request_log_checks.py                 # invalid responses, rejected requests, trial-1 check (Supplementary S7.1)
    python3 pilot_common_range.py                 # pilot: common-range Friedman tests (other pilot statistics: scripts/analyze.py)
    python3 pilot_invalid_recoding.py             # pilot: invalid responses recoded as a category (Supplementary S1.3)
    python3 make_figures_r1.py                    # Figures 1-3 (PNG 600 dpi, TIFF CMYK 300 dpi)
    python3 verify_item_selection.py              # optional: regenerates the item pool from the official PDFs (S8)

Results are written to `outputs/` (`revision_results.json`, `sensitivity_subsets.json`,
`consistently_wrong_by_temperature.json`, `modal_tie_check.json`, `request_log_checks.json`,
`item_selection_regenerated.json`).
Bootstrap work in `revision_analyses.py` is split into chunks that run in parallel and are cached in
`outputs/cache/`, so an interrupted run resumes where it stopped; `verify_item_selection.py` skips PDFs that were
already downloaded. Progress is printed.

## What verify_item_selection.py does
1. Runs the repository's `exp5_build_item_pool.py` unchanged in `outputs/poolregen/` (downloads the main booklets and
   answer keys of the 106th–115th examinations from the Ministry of Health, Labour and Welfare and builds the
   candidate pool).
2. Calls the original selection function `load_items()` of `exp5_run_full.py` on the regenerated pool (PyMuPDF text
   extraction; no API call).
3. Checks that the selected items, categories, keys, and prompt strings are identical to `exp5_prompts.csv` and writes
   the stage-wise counts reported in Supplementary Table S4c.
4. Rebuilds the review list of `exp5_select_items.py` (candidates that are Ethical/Legal or Priority under the
   first-pass or refined keywords) and checks that every entry of the exclusion lists is on it.

## Key conventions
* Valid response = a response containing an option number (its first digit 1–5 is parsed); the 11 responses without
  one are excluded. Consistency = 1 − H/log2(k) per (item, model, temperature) cell.
* Modal answer = most frequent valid answer; a tie is resolved to the answer occurring first in the data (pandas
  `value_counts().idxmax()`), which for the single tied cell is the answer of the first trial.
* Triage AUC: items ordered by ascending consistency; tied values handled by the expected capture curve under random
  ordering within ties (equivalent to counting ties as one half); per-temperature AUCs averaged; 2,000 bootstrap
  resamples of items drawn jointly across temperatures; temperatures without errors in a resample are omitted from
  that resample's average and resamples without any error are discarded.
* `extraction_audit_classification.csv` records the item-by-item prompt audit (clean / cosmetic / incomplete).
