# Nursing Consistency Index (NCI): A Reproducibility Audit of Three LLMs on Nursing Clinical Judgment

This repository contains the data, analysis scripts, and figure-generation
code for the study:

> Tajima, H. (2026). *Output consistency of large language models in nursing
> clinical judgement: a cross-model study introducing the Nursing Consistency
> Index, with cross-lingual external validation.* Manuscript submitted for
> publication.

The study comprises two experiments:

- **Study 1** — 30 author-written English nursing clinical-judgement scenarios
  (top-level `data/`, `scripts/`, `figures/`).
- **Study 2** — an independent cross-lingual external validation on 90 publicly
  released items of the Japanese National Nursing Examination
  (folder [`study2_japanese_national_exam/`](study2_japanese_national_exam/)).

## Overview

We evaluated the response consistency of three commercial large language
models — **GPT-4o** (OpenAI), **Claude Opus 4.5** (Anthropic), and
**Gemini 2.5 Flash-Lite** (Google).

**Study 1** queried 30 nursing clinical scenarios spanning three judgment
categories (Knowledge, Ethical, Priority) 30 times at each of four sampling
temperatures (0.0, 0.5, 1.0, 1.5). The Claude × T = 1.5 cell was rejected by the
Anthropic API because Anthropic does not accept temperature > 1.0, leaving 11
model–temperature conditions (9,900 attempted calls; 9,836 valid responses).

**Study 2** re-runs the same design on 90 publicly released items of the
Japanese National Nursing Examination (厚生労働省 / MHLW; 106th–115th
examinations, 2017–2026): 990 cells (90 items × 11 model–temperature
conditions) × 30 trials = 29,700 analysed responses. See
[`study2_japanese_national_exam/README.md`](study2_japanese_national_exam/README.md).

The **Nursing Consistency Index (NCI)** is defined as
`NCI = 1 − H(R)/log₂(k)`, where `H(R)` is the Shannon entropy of the
empirical response distribution over the repeated trials and `k` is the number
of answer options for the item. NCI = 1 indicates perfect consistency; NCI = 0
indicates uniform random responding. NCI is reported jointly with accuracy.

## Repository structure

```
nci-nursing-llm/
├── README.md                      # this file
├── LICENSE                        # MIT
├── CITATION.cff                   # citation metadata
├── requirements.txt               # Python dependencies
├── .gitignore
├── data/                          # Study 1 (English scenarios)
│   ├── scenarios.csv              # 30 clinical scenarios (stem + 4 options)
│   ├── raw_responses.csv          # 10,800 cleaned API responses
│   ├── nci_summary.csv            # per-cell NCI (330 cells)
│   └── cell_summary.csv           # per (model × temp) summary (11 cells)
├── scripts/                       # Study 1 analysis
│   ├── run_experiment.py          # API runner (template; needs API keys)
│   ├── analyze.py                 # reproduces all reported statistics
│   └── make_figures.py            # generates Figures 1–4
├── figures/                       # Study 1 figures
│   ├── Figure1_NCI_by_Temperature.{png,pdf}
│   ├── Figure2_NCI_by_Category.{png,pdf}
│   ├── Figure3_NCI_distribution.{png,pdf}
│   └── Figure4_NCI_vs_Accuracy.{png,pdf}
├── docs/
│   ├── data_dictionary.md         # column-by-column description
│   └── methodology.md             # study protocol summary
└── study2_japanese_national_exam/ # Study 2 (cross-lingual external validation)
    ├── README.md                  # Study 2 documentation
    ├── exp5_build_item_pool.py    # builds item pool from MHLW PDFs
    ├── exp5_select_items.py       # selects 90 items (30 per category)
    ├── exp5_run_full.py           # API runner (needs API keys)
    ├── exp5_analyze.py            # per-cell NCI/accuracy + summaries + figures
    ├── exp5_pool/exp5_prompts.csv # the 90 items used (stem, options, key, prompt)
    └── exp5_results/              # raw responses, per-cell metrics, summaries
```

## Quick start (Study 1)

### 1. Install dependencies

```bash
git clone https://github.com/aiprof202604-tech/nci-nursing-llm.git
cd nci-nursing-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce the published statistics

```bash
python scripts/analyze.py
```

This reads `data/raw_responses.csv` and prints all values reported in
the manuscript: per-cell mean NCI, Friedman χ² for temperature effects,
Kruskal–Wallis H for judgment-category effects, Fleiss' κ as the primary
convergent-validity measure, ICC(2,1) on a balanced 30 × 22 sub-sample as
a cross-reference, and per-model NCI–accuracy correlations.

### 3. Regenerate the figures

```bash
python scripts/make_figures.py
```

Outputs go to `figures/` (PNG at 600 dpi and PDF). All figures use a
colorblind-safe palette (Wong 2011).

### 4. Re-run the full experiment (optional, ~5–8 hours)

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
python scripts/run_experiment.py --workers 8
```

The `--dry-run` flag prints the call plan without making API requests.
For Study 2, see the instructions in
[`study2_japanese_national_exam/README.md`](study2_japanese_national_exam/README.md).

## Headline results (Study 1)

| Model | T = 0.0 | T = 0.5 | T = 1.0 | T = 1.5 |
|---|---|---|---|---|
| GPT-4o                | 0.985 (0.080) | 0.970 (0.101) | 0.950 (0.139) | 0.930 (0.176) |
| Claude Opus 4.5       | 1.000 (0.000) | 0.996 (0.019) | 0.984 (0.062) | — (API rejected) |
| Gemini 2.5 Flash-Lite | 1.000 (0.000) | 0.996 (0.019) | 0.991 (0.052) | 0.967 (0.076) |

Mean NCI (SD) per cell, n = 30 scenarios per cell.

- Temperature significantly degrades consistency for GPT-4o
  (Friedman χ²(3) = 13.36, p = .004) and Gemini (χ²(3) = 13.50, p = .004);
  not significant for Claude across the three feasible temperatures
  (χ²(2) = 3.71, p = .156).
- Judgment-category omnibus on scenario-level means is **not** significant
  (Kruskal–Wallis H(2) = 3.61, p = .16); descriptive ordering is
  Knowledge ≈ Priority > Ethical, driven by a small number of
  "reliably wrong" ethical scenarios in GPT-4o and Gemini.
- NCI correlates very strongly with Fleiss' κ across the 11 cells
  (Pearson r = .996, Spearman ρ = .98), supporting convergent validity.
- NCI–accuracy correlations are model-specific: Pearson r = .50
  (Spearman ρ = .82) for GPT-4o, r = .99 (ρ = 1.00) for Claude, and
  r = .07 (n.s.; ρ = .77) for Gemini; the near-zero Gemini Pearson value is
  produced by a few high-consistency/low-accuracy cells.

Study 2 reproduces the central findings on the Japanese examination items; see
[`study2_japanese_national_exam/`](study2_japanese_national_exam/).
See `docs/methodology.md` for the full Study 1 protocol and
`data/scenarios.csv` for the 30 scenarios.

## Data integrity notes

- **Claude × T = 1.5 (900 rows):** all rejected by the Anthropic API
  with `400 invalid_request_error: temperature: range: 0..1`. These rows
  are retained in `raw_responses.csv` with `response = API_REJECTED_T_OUT_OF_RANGE`
  for transparency. They do **not** enter any NCI or statistical computation.
- **62 additional API rate-limit errors** (49 Claude, 13 Gemini) are
  marked `API_RATE_LIMITED`. Two GPT-4o responses had non-conforming
  format and are marked `INVALID_FORMAT`. The 9,836 valid responses
  represent 99.4% of the 9,900 attempted calls.
- All `request_id` values from the original raw error messages have been
  stripped to keep the public file free of vendor-specific telemetry.
- **Study 2 row counts:** the runner log
  (`study2_japanese_national_exam/exp5_results/raw.csv`) includes resume/retry
  duplicate rows; `exp5_analyze.py` de-duplicates on
  `(item, model, temperature, trial)` to exactly 30 trials per cell
  (29,700 analysed responses). See the Study 2 README for details.

## Citation

If you use this dataset or code, please cite the article and this repository.

```bibtex
@article{tajima2026nci,
  title   = {Output consistency of large language models in nursing clinical
             judgement: a cross-model study introducing the Nursing
             Consistency Index, with cross-lingual external validation},
  author  = {Tajima, Hiroyuki},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

A `CITATION.cff` file is included for GitHub's "Cite this repository"
button, with the Zenodo-issued DOI (10.5281/zenodo.19918931).

## Licence

- **Code** (`scripts/`, `study2_japanese_national_exam/*.py`): MIT Licence.
- **Data** (`data/`, `figures/`, `study2_japanese_national_exam/exp5_*`):
  Creative Commons Attribution 4.0 International (CC BY 4.0).

The 30 Study 1 scenarios were drafted with the assistance of an LLM
(Claude Opus 4.7, Anthropic) and reviewed by the author — distinct from the
three models evaluated. See `docs/methodology.md` for details.

## Contact

Hiroyuki Tajima · Faculty of Nursing, Shumei University · Japan
