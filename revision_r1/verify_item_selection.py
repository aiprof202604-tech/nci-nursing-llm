#!/usr/bin/env python3
"""Regenerate the candidate pool from the official MHLW PDFs with the ARCHIVED pool builder and re-run the ORIGINAL
selection function, then report stage-wise counts and compare the selected items with those actually used.

Steps
  1. Runs study2_japanese_national_exam/exp5_build_item_pool.py of the repository unchanged, in NCI_OUT/poolregen
     (downloads the main booklets and answer keys of the 106th-115th examinations from www.mhlw.go.jp; files already
     downloaded are skipped, so an interrupted run resumes; the builder prints its progress).
  2. Imports exp5_run_full.py of the repository and calls its load_items() on the regenerated pool (PyMuPDF text
     extraction; NO API call is made).
  3. Compares the selected items with exp5_pool/exp5_prompts.csv (item set, category, key, prompt string) and writes
     the stage-wise counts to NCI_OUT/item_selection_regenerated.json.
  4. Rebuilds the review list of exp5_select_items.py (candidates that are Ethical/Legal or Priority under the
     first-pass or the refined keywords) and checks that every entry of the exclusion lists is on it.

Requirements: network access to www.mhlw.go.jp, pdfplumber (pool builder) and pymupdf (selection).
Environment: NCI_REPO (clone of aiprof202604-tech/nci-nursing-llm; default ./nci-nursing-llm), NCI_OUT (default ./outputs).
"""
import os, sys, csv, json, subprocess, importlib.util, collections, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

REPO = Path(os.environ.get("NCI_REPO", "nci-nursing-llm")).resolve()
S2 = REPO / "study2_japanese_national_exam"
OUT = Path(os.environ.get("NCI_OUT", "outputs")).resolve()
WORK = OUT / "poolregen"; WORK.mkdir(parents=True, exist_ok=True)

def main():
    pool_csv = WORK / "exp5_pool" / "exp5_candidate_pool.csv"
    if not pool_csv.exists():
        print("[1/4] regenerating the candidate pool with the archived builder (downloads official PDFs) ...", flush=True)
        subprocess.run([sys.executable, str(S2 / "exp5_build_item_pool.py")], cwd=WORK, check=True)
    else:
        print("[1/4] regenerated pool found; skipping download (delete poolregen/ to rebuild)", flush=True)

    print("[2/4] running the original selection function load_items() ...", flush=True)
    os.chdir(WORK)                                    # the run script uses relative paths (exp5_pool/...)
    spec = importlib.util.spec_from_file_location("exp5_run_full", S2 / "exp5_run_full.py")
    rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
    items, report = rf.load_items()

    print("[3/4] comparing with the items actually used and counting stages ...", flush=True)
    used = {r["item_uid"]: r for r in csv.DictReader(open(S2 / "exp5_pool" / "exp5_prompts.csv", encoding="utf-8-sig"))}
    got = {it["item_uid"]: it for it in items}
    mism = sorted(u for u in got if u in used and (got[u]["category"] != used[u]["category"]
                  or got[u]["key_letter"] != used[u]["key_letter"] or got[u]["prompt"] != used[u]["prompt"]))
    pool = list(csv.DictReader(open(pool_csv, encoding="utf-8-sig")))
    inc = [r for r in pool if r["include_candidate"] == "Y"]
    cat = collections.Counter(r["category_guess"] for r in inc)
    eth_removed = sorted(r["item_uid"] for r in inc if r["category_guess"] == "Ethical?" and r["item_uid"] in rf.ETH_EXCLUDE)
    pri_removed = sorted(r["item_uid"] for r in inc if r["category_guess"] == "Priority?" and r["item_uid"] in rf.PRI_EXCLUDE)
    byid = {r["item_uid"]: r for r in pool}
    pri_not_candidates = {u: (byid[u]["category_guess"], byid[u]["include_candidate"]) if u in byid else None
                          for u in sorted(rf.PRI_EXCLUDE) if u not in pri_removed}
    eth_not_candidates = {u: (byid[u]["category_guess"], byid[u]["include_candidate"]) if u in byid else None
                          for u in sorted(rf.ETH_EXCLUDE) if u not in eth_removed}
    res = dict(
        booklet_items=2400, segmented=len(pool),
        key_parsed=sum(1 for r in pool if r["official_key_num"].strip()),
        single_answer=sum(1 for r in pool if r["single_best"] == "Y"),
        single_answer_with_image_keyword=sum(1 for r in pool if r["single_best"] == "Y" and r["image_dependent"] == "Y"),
        candidates=len(inc), candidates_by_keyword_category=dict(cat),
        exclusion_list_sizes=dict(ethical=len(rf.ETH_EXCLUDE), priority=len(rf.PRI_EXCLUDE)),
        removed=dict(ethical=len(eth_removed), priority=len(pri_removed)),
        exclusion_entries_not_candidates_of_that_category=dict(ethical=eth_not_candidates, priority=pri_not_candidates),
        selection_report=[dict(category=c, selected=n, candidates_after_review=p, skipped=s) for c, n, p, s in report],
        selected_now=len(got), used_in_study=len(used), same_item_set=set(got) == set(used),
        items_with_different_category_key_or_prompt=mism,
        segmented_by_examination=dict(collections.Counter(r["year"] for r in pool)),
    )
    print("[4/4] rebuilding the review list of exp5_select_items.py (pdfplumber text) ...", flush=True)
    spec2 = importlib.util.spec_from_file_location("exp5_select_items", S2 / "exp5_select_items.py")
    si = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(si)
    stems = {}
    for y in sorted({r["year"] for r in inc}):
        for sess in ("AM", "PM"):
            for q, seg in si.segment(si.extract_pages(WORK / "exp5_pool" / y / f"{sess}_main.pdf")).items():
                stems[(y, sess, int(q))] = seg
        print(f"    extracted {y}", flush=True)
    rv = []
    for r in inc:
        seg = stems.get((r["year"], r["session"], int(r["q_number"])), "")
        rv.append(dict(uid=r["item_uid"], first=r["category_guess"].rstrip("?"), refined=si.refined_cat(seg) if seg else "Knowledge"))
    eth_rows = [x for x in rv if "Ethical" in (x["first"], x["refined"])]
    pri_rows = [x for x in rv if "Priority" in (x["first"], x["refined"])]
    res["review_list"] = dict(
        ethical_rows=len(eth_rows), priority_rows=len(pri_rows),
        distinct_items=len({x["uid"] for x in eth_rows} | {x["uid"] for x in pri_rows}),
        priority_rows_refined_only=sorted(x["uid"] for x in pri_rows if x["first"] != "Priority"),
        ethical_rows_refined_only=sorted(x["uid"] for x in eth_rows if x["first"] != "Ethical"),
        eth_exclude_on_list=sum(x["uid"] in rf.ETH_EXCLUDE for x in eth_rows),
        pri_exclude_on_list=sum(x["uid"] in rf.PRI_EXCLUDE for x in pri_rows))
    for k, v in res.items(): print(f"  {k}: {v}")
    json.dump(res, open(OUT / "item_selection_regenerated.json", "w"), indent=1, ensure_ascii=False)
    print("written", OUT / "item_selection_regenerated.json")
    if not (res["same_item_set"] and not mism):
        sys.exit("[!] the regenerated selection differs from the items used in the study")

if __name__ == "__main__":
    main()
