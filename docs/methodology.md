# Methodology

This document summarises the study protocol. For the complete account
including statistical assumptions, limitations, and discussion, see the
manuscript.

## Models evaluated

| Short key | Full API identifier | Vendor |
|---|---|---|
| `gpt`    | `gpt-4o`                | OpenAI |
| `claude` | `claude-opus-4-5`       | Anthropic |
| `gemini` | `gemini-2.5-flash-lite` | Google |

All three models were accessed through standard paid commercial APIs.
The author has no financial or institutional relationship with any of
these vendors.

A separate model — Claude Opus 4.7 (Anthropic) — was used as a drafting
assistant for the 30 clinical scenarios. The author reviewed and revised
all scenarios and answer keys before the experiment. **The drafting
model is distinct from the three models evaluated and was not used at
data-collection time.**

## Scenarios

30 four-option multiple-choice scenarios, evenly distributed across
three judgment categories (10 each):

- **Knowledge:** Factual nursing knowledge (anatomy, pharmacology,
  pathophysiology, infection control, basic procedures).
- **Ethical:** Ethically loaded clinical situations (confidentiality,
  autonomy, end-of-life decisions, resource allocation, mandatory
  reporting).
- **Priority:** Triage and prioritisation under finite resources or
  time pressure.

Scenarios were tagged against NANDA-I 13th edition domains
(Herdman et al., 2024) for descriptive coverage. The full set is in
`data/scenarios.csv`.

## Experimental design

- **Planned design:** 3 models × 4 temperatures × 30 scenarios × 30
  trials = 10,800 API calls.
- **Sampling temperatures:** 0.0, 0.5, 1.0, 1.5.
- **Top-p:** Default for each vendor (not modified). Rationale:
  the audit targets *out-of-the-box* behaviour as encountered by
  educators using these systems through standard interfaces.
- **System prompt:** Fixed across all calls, instructing the model to
  respond with exactly one letter (`A`, `B`, `C`, or `D`).
- **`max_tokens`:** 5 — sufficient for a single letter, conservative
  enough to fail-fast on verbose responses.
- **One cell unavailable:** Claude × T = 1.5. The Anthropic API
  rejects temperature > 1.0 with `400 invalid_request_error: temperature: range: 0..1`.
  All 900 rows are retained in `raw_responses.csv` for transparency
  but excluded from NCI and statistical computations.

## Outcome: the Nursing Consistency Index (NCI)

For a cell with `n` valid responses distributed over `k = 4` options,
let `R` be the empirical response distribution. Then:

```
H(R) = − Σ pᵢ · log₂(pᵢ)        # Shannon entropy
NCI  = 1 − H(R) / log₂(k)
```

- `NCI = 1` → all `n` responses identical (zero entropy).
- `NCI = 0` → uniform distribution across all `k` options
  (maximum entropy).

NCI is bounded in [0, 1] and is independent of which option dominates,
so a model that is *reliably wrong* on a given scenario receives the
same NCI as a model that is *reliably right*. NCI therefore measures
**stability**, not correctness — accuracy is reported separately.

### Finite-`n` caveat

NCI is biased upward at small `n` because rare options are likely to
be missed. With `n = 30` per cell this bias is small but nonzero.
We report a sensitivity analysis using only the first 22 valid trials
per cell in the manuscript; conclusions are unchanged.

## Statistical analyses

All analyses use `scipy.stats` (see `scripts/analyze.py`).

| Question | Test | Notes |
|---|---|---|
| Does temperature affect NCI within a model? | Friedman χ² with scenarios as blocks | Computed separately for each model. Claude uses 3 temperatures (T = 0–1.0); GPT-4o and Gemini use 4. |
| Does judgment category affect NCI? | Kruskal–Wallis H on **scenario-level mean NCI** (n = 10 per category) | Pooled across models and temperatures to avoid pseudo-replication. Omnibus *not* significant in our data; descriptive ordering only. |
| Does NCI converge with established reliability indices? | Fleiss' κ on *all 9,836 valid responses* per cell, then correlated with mean NCI across the 11 cells (Pearson, Spearman) | Primary convergent-validity measure. ICC(2,1) on a balanced 30 × 22 sub-sample is reported as a cross-reference. |
| Is NCI separable from accuracy? | Pearson correlation between cell-level NCI and accuracy, computed *per model* | Strongly model-specific: r = .50 (GPT-4o), r = .99 (Claude), r = .07 (Gemini). |

A descriptive Stouffer's Z combination of Friedman tests across models
is reported for completeness; it is not interpreted confirmatorily.

## What this study does NOT do

- It does not establish *correctness* of the answer keys for ethical
  scenarios. Some scenarios (notably B-03, adolescent confidentiality)
  have jurisdiction-dependent answers; the answer key reflects the
  framework used by the author and not a universal truth.
- It does not test whether one model is "better" than another for
  clinical use. NCI characterises *stability*; clinical use additionally
  requires accuracy, calibration, and human-in-the-loop oversight.
- It does not evaluate fine-tuned, RAG-augmented, or chain-of-thought
  variants. Results apply to the default API behaviour of the three
  named models at the time of data collection.
- It does not generalise beyond 4-option multiple-choice format.

## Reproducibility checklist

- [x] Model identifiers fully specified (`gpt-4o`, `claude-opus-4-5`,
      `gemini-2.5-flash-lite`).
- [x] System prompt and per-call parameters fixed across all calls.
- [x] All 10,800 raw responses preserved (`data/raw_responses.csv`).
- [x] Error rows transparently labelled and excluded from analysis.
- [x] Analysis script reproduces every reported number
      (`scripts/analyze.py`).
- [x] Figure script reproduces every figure (`scripts/make_figures.py`).
- [x] Data and code released under permissive licences (CC BY 4.0,
      MIT).
- [ ] Permanent DOI via Zenodo *(set after first GitHub release)*.
