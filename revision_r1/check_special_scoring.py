#!/usr/bin/env python3
"""Supplementary S8.1: were any of the 90 selected items withdrawn or scored specially by the Ministry?
Downloads the Ministry of Health, Labour and Welfare pass-announcement pages of the 106th-115th national nursing
examinations (2017-2026, the screening universe), extracts the items listed under "items for which scoring exclusion or other special
treatment was applied" (採点除外等の取扱いをした問題) for the nurse examination, and checks them against the 90
selected items and, if the regenerated candidate pool of verify_item_selection.py is present
(outputs/poolregen/exp5_pool/exp5_candidate_pool.csv), against every screened item and the 1,193 candidates. Downloads run in 4 threads, already downloaded pages are reused (resume), and progress is printed.
Needs network access to www.mhlw.go.jp. No API call to any language model.
Environment: NCI_REPO (default ./nci-nursing-llm), NCI_OUT (default ./outputs). Output: outputs/special_scoring.json."""
import html, json, os, re, subprocess, unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

REPO = Path(os.environ.get("NCI_REPO", "nci-nursing-llm")).resolve()
OUT = Path(os.environ.get("NCI_OUT", "outputs")).resolve()
PAGES = OUT / "mhlw_pass_pages"
EXAM = {2017: 106, 2018: 107, 2019: 108, 2020: 109, 2021: 110, 2022: 111, 2023: 112, 2024: 113, 2025: 114, 2026: 115}


def fetch(year):
    f = PAGES / f"about_{year}.html"
    if f.exists():
        return year, "cached"
    url = f"https://www.mhlw.go.jp/general/sikaku/successlist/{year}/siken03_04_05/about.html"
    r = subprocess.run(["curl", "-sS", "-m", "60", "-o", str(f), "-w", "%{http_code}", url], capture_output=True, text=True)
    if r.stdout != "200":
        f.unlink(missing_ok=True)
    return year, r.stdout


def special_items(year):
    raw = (PAGES / f"about_{year}.html").read_bytes()
    for enc in ("utf-8", "cp932"):
        try:
            t = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    t = unicodedata.normalize("NFKC", re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t))))
    i = t.find("看護師国家試験における")
    if i < 0:
        return []                                   # no special-scoring notice for the nurse examination
    seg = t[i:i + 1500].split("(参考2)")[0].split("照会先")[0]
    return [("AM" if s == "午前" else "PM") + f"-{int(n)}" for s, n in re.findall(r"(午前|午後)問題 第 ?(\d+) ?問", seg)]


def main():
    PAGES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        for k, (y, s) in enumerate(ex.map(fetch, EXAM), 1):
            print(f"[{k}/{len(EXAM)}] {y} ({EXAM[y]}th): {s}", flush=True)
    special = {y: special_items(y) for y in EXAM}
    pr = pd.read_csv(REPO / "study2_japanese_national_exam" / "exp5_pool" / "exp5_prompts.csv", encoding="utf-8-sig")
    hits = []
    for uid in pr.item_uid:
        e, y, s, n = re.match(r"(\d+)_(\d+)-(AM|PM)-(\d+)", uid).groups()
        if f"{s}-{int(n)}" in special.get(int(y), []):
            hits.append(uid)
    pool_f = OUT / "poolregen" / "exp5_pool" / "exp5_candidate_pool.csv"
    pool_hits = None
    if pool_f.exists():
        pool = pd.read_csv(pool_f, encoding="utf-8-sig", dtype=str).fillna("")
        pool_hits = []
        for _, r in pool.iterrows():
            e, y, s, n = re.match(r"(\d+)_(\d+)-(AM|PM)-(\d+)", r.item_uid).groups()
            if f"{s}-{int(n)}" in special.get(int(y), []):
                pool_hits.append(dict(item_uid=r.item_uid, single_best=r.single_best, candidate=r.include_candidate,
                                      category_guess=r.category_guess))
        cand = [h for h in pool_hits if h["candidate"] == "Y"]
        print(f"specially scored items in the screened pool: {len(pool_hits)}; among the candidates: {len(cand)}")
        for h in cand:
            print("   candidate:", h["item_uid"], h["category_guess"])
    else:
        print("(candidate pool not found; run verify_item_selection.py first to check the candidates)")
    for y in EXAM:
        print(f"  {EXAM[y]}th ({y}): {len(special[y])} items with special scoring: {', '.join(special[y]) or 'none'}")
    print(f"selected items ({len(pr)}) with special scoring: {hits or 'none'}")
    (OUT / "special_scoring.json").write_text(json.dumps(dict(special={str(EXAM[y]): v for y, v in special.items()},
                                                              selected_hits=hits, pool_hits=pool_hits), indent=1))


if __name__ == "__main__":
    main()
