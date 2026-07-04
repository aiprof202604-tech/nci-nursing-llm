# LLM response consistency on nursing clinical judgement — data, analysis code, and figures

This repository contains the data, analysis scripts, and figure-generation
code for the study:

> Tajima, H. (2026). *When consistency stops tracking correctness: a
> deployment-safety evaluation of large language models on a national nursing
> licensing examination.* Manuscript submitted for publication.

The study comprises two experiments. In the current manuscript, **Study 2**
(the Japanese national examination) is the **primary** analysis and **Study 1**
(the English scenarios) is reported as **supplementary**. The repository layout
retains Study 1 at the top level for historical reasons.

- **Study 2 (primary)** — an independent, externally authored, cross-lingual
  external validation on 90 publicly released items of the Japanese National
  Nursing Examination
  (folder [`study2_japanese_national_exam/`](study2_japanese_national_exam/)).
- **Study 1 (supplementary)** — 30 author-written English nursing
  clinical-judgement scenarios
  (top-level `data/`, `scripts/`, `figures/`).

## Overview

We evaluated the response consistency of three commercial large language
models — **GPT-4o** (OpenAI), **Claude Opus 4.5** (Anthropic), and
**Gemini 2.5 Flash-Lite** (Google) — as a deployment-safety question: whether
output consistency can serve as a pre-deployment safety signal for
clinical-judgement items.

**Study 2 (primary)** re-runs the Study 1 design on 90 publicly released items
of the Japanese National Nursing Examination (厚生労働省 / MHLW; 106th–115th
examinations, 2017–2026): 990 cells (90 items × 11 model–temperature
conditions) × 30 trials = 29,700 analysed responses. See
[`study2_japanese_national_exam/README.md`](study2_japanese_national_exam/README.md).

**Study 1 (supplementary)** queried 30 nursing clinical scenarios spanning three
judgment categories (Knowledge, Ethical, Priority) 30 times at each of four
sampling temperatures (0.0, 0.5, 1.0, 1.5). The Claude × T = 1.5 cell was
rejected by the Anthropic API because Anthropic does not accept temperature
> 1.0, leaving 11 model–temperature conditions (9,900 attempted calls; 9,836
valid responses).

Response consistency is measured as
`C = 1 − H(R)/log₂(k)`, the normalised complement of the Shannon entropy
`H(R)` of the empirical response distribution over the repeated trials, where
`k` is the number of answer options for the item. A value of 1 indicates
perfect consistency; 0 indicates uniform random responding. Consistency is
reported jointly with accuracy. (In the source code and file names this
quantity is abbreviated `NCI`; it is a per-item measure, not a claim of a new
psychometric instrument.)

## Repository structure
