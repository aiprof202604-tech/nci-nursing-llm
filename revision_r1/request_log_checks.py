#!/usr/bin/env python3
"""Checks on the request log and on the analyzed responses (Supplementary S7): responses without an option number,
the Claude requests rejected for exhausted credit and the successful requests to the other models logged among them,
and whether trial 1 (collected in an initial smoke-test run) behaves like trials 2-30. Reads only the public raw data;
no API calls. Environment: NCI_REPO (default ./nci-nursing-llm), NCI_OUT (default ./outputs)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import revision_analyses as ra

def main():
    out = {}
    raw = pd.read_csv(ra.RAW_DEDUP, encoding="utf-8-sig")
    raw["ans"] = raw.parsed_letter.fillna("").astype(str).str.strip()
    inv = raw[raw.ans == ""]
    out["invalid_responses"] = [dict(item=r.item_uid, model=r.model, temp=float(r.temp), trial=int(r.trial),
                                     raw_response=str(r.raw_response)) for r in inv.itertuples()]
    print(f"[1/3] responses without an option number: {len(inv)}")
    for d in out["invalid_responses"]: print("     ", d)

    lg = pd.read_csv(ra.RAW_LOG, encoding="utf-8-sig")
    err = lg.raw_response.astype(str).str.startswith("__ERROR__").values
    pos = np.flatnonzero(err)
    win = lg.iloc[pos.min():pos.max() + 1]; werr = err[pos.min():pos.max() + 1]
    run = best = 0
    for e in err:
        run = run + 1 if e else 0; best = max(best, run)
    out.update(log_rows=int(len(lg)), credit_errors=int(err.sum()),
               credit_error_models=lg[err].model.value_counts().to_dict(),
               credit_error_messages_with_credit_text=int(lg[err].raw_response.str.contains("credit balance is too low").sum()),
               error_window_rows=int(len(win)), other_rows_in_error_window=int((~werr).sum()),
               other_rows_in_error_window_by_model=win[~werr].model.value_counts().to_dict(),
               longest_consecutive_error_run=int(best),
               trial1_rows_at_top_of_log=bool((lg.iloc[:990].trial == 1).all()))
    print(f"[2/3] request log: {out['credit_errors']} rejected requests ({out['credit_error_models']}), "
          f"{out['other_rows_in_error_window']} successful rows of other models among them "
          f"({out['other_rows_in_error_window_by_model']}); longest run {best}; trial-1 rows first: {out['trial1_rows_at_top_of_log']}")

    agree = {t: [] for t in range(1, 31)}; acc = {t: [] for t in range(1, 31)}
    groups = list(raw.groupby(["item_uid", "model", "temp"]))
    for n, (_, g) in enumerate(groups, 1):
        g = g.set_index("trial")
        for t in range(1, 31):
            others = g.drop(index=t).ans; others = others[others != ""]
            agree[t].append(int(g.loc[t, "ans"] == others.value_counts().idxmax()))
            acc[t].append(int(g.loc[t, "correct"]))
        if n % 200 == 0 or n == len(groups): print(f"\r[3/3] trial-1 check: {n}/{len(groups)} cells", end="", flush=True)
    print()
    A = np.array([np.mean(agree[t]) for t in range(1, 31)]); C = np.array([np.mean(acc[t]) for t in range(1, 31)])
    t0 = [g for (it, m, tp), g in raw[raw.temp == 0].groupby(["item_uid", "model", "temp"])]
    n_un = n_dis = 0
    for g in t0:
        g = g.set_index("trial"); rest = g.drop(index=1).ans
        if rest.nunique() == 1:
            n_un += 1; n_dis += int(g.loc[1, "ans"] != rest.iloc[0])
    out.update(trial1_agree=float(A[0]), others_agree_min=float(A[1:].min()), others_agree_max=float(A[1:].max()),
               trial1_acc=float(C[0]), others_acc_min=float(C[1:].min()), others_acc_max=float(C[1:].max()),
               t0_unanimous_cells=n_un, t0_trial1_disagreements=n_dis)
    print(f"      trial 1 agreement {A[0]:.4f} (others {A[1:].min():.4f}-{A[1:].max():.4f}); accuracy {C[0]:.4f} "
          f"(others {C[1:].min():.4f}-{C[1:].max():.4f}); T=0 unanimous cells {n_un}, trial-1 disagreements {n_dis}")
    json.dump(out, open(ra.OUT / "request_log_checks.json", "w"), indent=1, ensure_ascii=False)
    print("written", ra.OUT / "request_log_checks.json")

if __name__ == "__main__":
    main()
