"""
NCI experiment runner.

Calls 3 commercial LLMs (GPT-4o, Claude Opus 4.5, Gemini 2.5 Flash-Lite)
for 30 scenarios x 4 temperatures x 30 trials = 10,800 calls.

Note: Anthropic API rejects temperature > 1.0; the Claude T = 1.5 cell
will yield 900 API errors (this is expected behaviour, not a bug).

Set environment variables before running:
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    export GEMINI_API_KEY=...

Usage:
    python scripts/run_experiment.py            # full run (~5-8 hours)
    python scripts/run_experiment.py --dry-run  # show plan only

Requires: openai, anthropic, requests, pandas, openpyxl
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SYSTEM_PROMPT = (
    "You are taking a nursing clinical judgment exam. "
    "For each question, respond with ONLY a single letter: A, B, C, or D. "
    "Do not provide any explanation, reasoning, or additional text. "
    "Your response must be exactly one character: A, B, C, or D."
)

MODELS = [
    {"key": "gpt",    "vendor": "openai",    "name": "gpt-4o",                "temps": [0.0, 0.5, 1.0, 1.5]},
    {"key": "claude", "vendor": "anthropic", "name": "claude-opus-4-5",       "temps": [0.0, 0.5, 1.0, 1.5]},
    {"key": "gemini", "vendor": "google",    "name": "gemini-2.5-flash-lite", "temps": [0.0, 0.5, 1.0, 1.5]},
]

N_TRIALS = 30


def call_openai(model_name, scenario_text, temperature):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": scenario_text},
        ],
        temperature=temperature,
        max_tokens=5,
    )
    return resp.choices[0].message.content.strip()


def call_anthropic(model_name, scenario_text, temperature):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model_name,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": scenario_text}],
        temperature=temperature,
        max_tokens=5,
    )
    return resp.content[0].text.strip()


def call_gemini(model_name, scenario_text, temperature):
    import requests
    api_key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": scenario_text}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 5},
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


CALLERS = {"openai": call_openai, "anthropic": call_anthropic, "google": call_gemini}


def build_user_prompt(row):
    return (
        f"{row['scenario']}\n\n"
        f"A. {row['option_A']}\n"
        f"B. {row['option_B']}\n"
        f"C. {row['option_C']}\n"
        f"D. {row['option_D']}\n\n"
        f"Answer:"
    )


def run_one(spec):
    """spec: dict with all info needed for one API call."""
    model = spec["model"]
    try:
        ans = CALLERS[model["vendor"]](model["name"], spec["prompt"], spec["temperature"])
    except Exception as exc:
        ans = f"ERROR: {exc}"
    return {
        "question_id": spec["question_id"],
        "category": spec["category"],
        "model": model["key"],
        "model_name": model["name"],
        "temperature": spec["temperature"],
        "trial": spec["trial"],
        "answer": ans,
        "intended_answer": spec["intended_answer"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Show plan without calling APIs")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel API workers per model")
    ap.add_argument("--out", default=str(DATA / "raw_responses_new.csv"))
    args = ap.parse_args()

    scenarios = pd.read_csv(DATA / "scenarios.csv")
    specs = []
    for _, row in scenarios.iterrows():
        prompt = build_user_prompt(row)
        for model in MODELS:
            for temp in model["temps"]:
                for trial in range(1, N_TRIALS + 1):
                    specs.append({
                        "question_id": row["question_id"],
                        "category": row["category"],
                        "intended_answer": row["intended_answer"],
                        "model": model,
                        "temperature": temp,
                        "trial": trial,
                        "prompt": prompt,
                    })

    print(f"Total planned API calls: {len(specs):,}")
    print(f"Output file: {args.out}")
    if args.dry_run:
        print("Dry-run mode; no API calls made.")
        return

    for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        if not os.environ.get(var):
            print(f"ERROR: ${var} not set.", file=sys.stderr)
            sys.exit(1)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, s): i for i, s in enumerate(specs)}
        for k, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if k % 100 == 0:
                elapsed = time.time() - t0
                rate = k / elapsed
                eta = (len(specs) - k) / rate / 60
                print(f"  [{k:>5}/{len(specs)}] {rate:.1f} calls/s, ETA {eta:.0f} min")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
