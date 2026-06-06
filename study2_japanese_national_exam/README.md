# Study 2 — Cross-lingual external validation (Japanese National Nursing Examination)

This folder contains the data and code for **Study 2** of the manuscript
*"Output consistency of large language models in nursing clinical judgement: a
cross-model study introducing the Nursing Consistency Index, with cross-lingual
external validation."* In the source code Study 2 is named **experiment 5
(`exp5`)**.

Study 2 re-runs the Study 1 design on **90 publicly released items of the
Japanese National Nursing Examination** (厚生労働省 / MHLW; 106th–115th
examinations, 2017–2026), as an independent, externally authored, non-English
replication. The raw examination PDFs are public and are available from the
MHLW examination page cited in the manuscript.

## What was run

- **Models:** `gpt-4o` (OpenAI), `claude-opus-4.5` (Anthropic),
  `gemini-2.5-flash-lite` (Google). Accessed June 2026.
- **Sampling temperatures:** GPT-4o and Gemini at {0.0, 0.5, 1.0, 1.5}; Claude at
  {0.0, 0.5, 1.0} only, because the Anthropic API rejects temperature > 1.0.
- **Trials:** 30 per cell, where a *cell* = (item × model × temperature).
- **Scale:** 90 items × 11 model–temperature conditions = **990 cells**, i.e.
  **29,700 analysed responses**. The run completed with zero API errors;
  11 responses (0.037%) were empty/unparseable.
- **NCI** = 1 − H(R) / log₂(k), where H(R) is the Shannon entropy (base 2) of the
  empirical response distribution over the 30 trials and k is the number of
  answer options for that item (4 or 5). Empty/unparseable responses are treated
  as a distinct category for NCI and as incorrect for accuracy.

## Item selection (`exp5_build_item_pool.py`, `exp5_select_items.py`)

Items are taken from the **main examination booklets only**: morning
(`-05a_01.pdf`) and afternoon (`-05c_01.pdf`); **re-examination booklets
(`05b`/`05d`) and supplementary picture booklets (`_02`) are excluded.** Text is
extracted with PyMuPDF (`get_text("text", sort=True)`) and NFKC-normalised.
Image-dependent items, multiple-answer items, items excluded from official
scoring, and context-dependent sub-questions are filtered out. The final pool is
**90 self-contained single-best-answer items: 30 Knowledge, 30 Priority,
30 Ethical/Legal.**

## Files

```
study2_japanese_national_exam/
├── exp5_build_item_pool.py     # downloads MHLW PDFs, extracts the candidate item pool
├── exp5_select_items.py        # selects 30 items per category (90 total)
├── exp5_run_full.py            # queries the 3 models × temperatures × 30 trials (API keys via env)
├── exp5_analyze.py             # computes NCI/accuracy per cell + summaries + figures
├── exp5_pool/
│   └── exp5_prompts.csv        # the 90 items actually used: stem, options, answer key, k, exact prompt
└── exp5_results/
    ├── raw.csv                 # complete runner log, as produced (see "Row counts" below)
    ├── raw_dedup_29700.csv     # analysis-ready: exactly 30 trials per cell (29,700 rows)
    ├── exp5_cell_metrics.csv   # per-cell NCI, entropy, accuracy, modal answer (990 rows)
    ├── exp5_summary_model_temp.csv
    ├── exp5_summary_model_category.csv
    ├── exp5_summary_model_cat_temp.csv
    ├── exp5_summary_recent.csv         # recent (2022–2026) vs older (2017–2021) strata
    ├── exp5_shared_blindspots.csv      # items where >1 vendor agrees on the same wrong answer
    ├── exp5_item_table.csv             # per-item per-model accuracy/NCI (supplementary)
    └── exp5_findings.md                # analysis notes
```

### Column dictionaries

- **`raw.csv` / `raw_dedup_29700.csv`:** `item_uid, category, model, temp, trial,
  raw_response, parsed_letter, key_letter, correct`
- **`exp5_prompts.csv`:** `item_uid, category, recent, k, key_letter, stem,
  options, prompt`
- **`exp5_cell_metrics.csv`:** `item_uid, category, recent, k, key, model, temp,
  n_trials, nci, entropy_bits, accuracy, modal_answer, modal_correct`

## Row counts: why `raw.csv` has 30,909 rows but the study reports 29,700

`raw.csv` is the **complete, unedited runner log**. Because the runner supports
resume/retry, it appended duplicate rows for ~42 Claude cells (the same trial
numbers written more than once), giving 30,909 logged rows. `exp5_analyze.py`
**de-duplicates on `(item_uid, model, temp, trial)` (keeping the last record)**
before computing any metric, yielding exactly **30 trials per cell = 29,700
unique analysed responses**, which is the number reported in the manuscript.
`raw_dedup_29700.csv` is that de-duplicated dataset, provided for convenience;
running `exp5_analyze.py` on `raw.csv` produces identical results.

## Reproduction (verified)

Running the analysis reproduces the manuscript exactly:

- **Table 3 (mean NCI by temperature):** GPT-4o 0.992 / 0.981 / 0.967 / 0.940;
  Gemini 1.000 / 0.987 / 0.951 / 0.932; Claude 1.000 / 0.999 / 0.996 — and the
  accompanying accuracy values — all match.
- **Per-cell:** all **990/990** cells in `exp5_cell_metrics.csv` are reproduced
  from the raw data.
- Key items in §3.9 reproduce, e.g. item `115_2026-AM-31` (GPT-4o accuracy
  0.017; Claude 1.000; Gemini 0.967) and item `114_2025-AM-75` (GPT-4o and Gemini
  converge on the same wrong option D; Claude answers correctly).

## How to run

```bash
pip install -r requirements.txt          # PyMuPDF, requests, openai, anthropic, google-generativeai, matplotlib
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...                 # (or GEMINI_API_KEY)

python exp5_build_item_pool.py            # build candidate pool from MHLW PDFs  -> exp5_pool/
python exp5_select_items.py               # select 90 items                      -> exp5_pool/
python exp5_run_full.py                   # query models (writes exp5_results/raw.csv)
python exp5_analyze.py                    # metrics, summaries, figures          -> exp5_results/
```

Model outputs are time-dependent because commercial models are updated; the
files here are the June 2026 snapshot used in the manuscript.
