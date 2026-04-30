# Data Dictionary

This file describes every column in every data file. All files are UTF-8
encoded CSV with comma separators and a single header row.

---

## `data/scenarios.csv`

The 30 clinical scenarios used in the study. One row per scenario.

| Column | Type | Description |
|---|---|---|
| `question_id` | string | Unique identifier. Format `<letter>-<NN>`. `A` = Knowledge, `B` = Ethical, `C` = Priority. |
| `category` | string | One of `Knowledge`, `Ethical`, `Priority`. |
| `scenario` | string | Free-text clinical vignette presented to the model. |
| `option_A` | string | Response option A. |
| `option_B` | string | Response option B. |
| `option_C` | string | Response option C. |
| `option_D` | string | Response option D. |
| `intended_answer` | string | The predefined answer key (one of `A`, `B`, `C`, `D`). For ethical scenarios this should be read as the *target* response under the framework used by the author rather than a uniquely "correct" answer. |
| `nanda_domain` | string | NANDA-I 13th edition domain tag (Herdman et al., 2024) for descriptive coverage. |

---

## `data/raw_responses.csv`

All 10,800 API responses (one row per call), cleaned and anonymised.

| Column | Type | Description |
|---|---|---|
| `question_id` | string | Foreign key to `scenarios.csv`. |
| `category` | string | Redundant with `scenarios.csv` for convenience. |
| `model` | string | Short model key: `gpt`, `claude`, or `gemini`. |
| `model_name` | string | Full API model identifier: `gpt-4o`, `claude-opus-4-5`, or `gemini-2.5-flash-lite`. |
| `temperature` | float | Sampling temperature: 0.0, 0.5, 1.0, or 1.5. |
| `trial` | int | Trial number within the cell, 1–30. |
| `response` | string | Cleaned response. Either a single letter `A`–`D` (valid) or one of the error codes below. |
| `intended_answer` | string | Copy of `scenarios.csv` field. |
| `correct` | int | 1 if `response == intended_answer`, else 0. (Always 0 for error rows.) |
| `is_valid` | bool | True iff `response ∈ {A, B, C, D}`. |

### Error codes in the `response` column

| Code | Count | Meaning |
|---|---|---|
| `API_REJECTED_T_OUT_OF_RANGE` | 900 | Anthropic API returned `400 invalid_request_error: temperature: range: 0..1` (entire Claude × T = 1.5 cell). Expected behaviour, not a bug. |
| `API_RATE_LIMITED` | 62 | HTTP 429 / `overloaded_error` / 529. |
| `INVALID_FORMAT` | 2 | Response was not a single letter. |

There are zero `API_TIMEOUT`, `API_OVERLOADED`, or `API_ERROR_OTHER`
rows in this dataset, but those codes are reserved by the cleaning script
for re-runs.

### Row count check

```
3 models × 4 temperatures × 30 scenarios × 30 trials = 10,800 rows
of which:
  9,836 valid (response in {A,B,C,D})
    900 Claude T=1.5 API rejections
     62 rate-limit errors (49 Claude T=0–1.0, 13 Gemini, 0 GPT)
      2 INVALID_FORMAT (GPT-4o)
```

---

## `data/nci_summary.csv`

One row per (model × temperature × scenario) cell — 330 rows
(3 × 4 × 30 minus 30 for the Claude T = 1.5 cell that was skipped in
this aggregation because it has no valid responses; in this file the
Claude T = 1.5 rows are simply absent).

| Column | Type | Description |
|---|---|---|
| `question_id` | string | Foreign key to `scenarios.csv`. |
| `category` | string | One of `Knowledge`, `Ethical`, `Priority`. |
| `model` | string | `gpt`, `claude`, or `gemini`. |
| `model_name` | string | Full API model identifier. |
| `temperature` | float | 0.0, 0.5, 1.0, or 1.5. |
| `n_valid` | int | Number of valid responses out of 30 trials. |
| `n_errors` | int | `30 − n_valid`. |
| `NCI` | float | Nursing Consistency Index for this cell, computed only on valid responses. Range [0, 1]. |
| `n_correct` | int | Number of valid responses matching `intended_answer`. |
| `accuracy` | float | `n_correct / n_valid`. |

---

## `data/cell_summary.csv`

One row per (model × temperature) — 11 rows (Claude T = 1.5 absent).

| Column | Type | Description |
|---|---|---|
| `model` | string | `gpt`, `claude`, or `gemini`. |
| `model_name` | string | Full API model identifier. |
| `temperature` | float | 0.0, 0.5, 1.0, or 1.5. |
| `n_scenarios` | int | Always 30. |
| `mean_NCI` | float | Mean NCI across the 30 scenarios. |
| `sd_NCI` | float | Standard deviation of NCI across scenarios. |
| `mean_accuracy` | float | Mean accuracy across scenarios. |
| `sd_accuracy` | float | Standard deviation of accuracy. |
| `total_valid_responses` | int | Sum of `n_valid` across the 30 scenarios. |
| `total_errors` | int | Sum of `n_errors` across the 30 scenarios. |
| `n_perfect_consistency` | int | Number of scenarios in this cell with `NCI ≥ 0.99999` (effectively 1.0). |
| `pct_perfect_consistency` | float | `n_perfect_consistency / 30 × 100`. |
