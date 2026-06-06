#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp5_build_item_pool.py  —  実験5（クロスリンガル再現）項目プール構築

厚生労働省が公開する看護師国家試験（第111〜115回）の公式PDFから、
- 午前本文 / 午後本文 / 正答（別冊・再試験は除外）
を自動取得・抽出し、
  (1) 公式正答との照合で「単一解 / 複数選択」を確定
  (2) 本文から「選択肢数(k)」「画像依存(別冊/心電図/エックス線/図/写真 等)」を判定
  (3) 倫理・優先のキーワードでカテゴリ候補を付与
して候補CSV（exp5_pool/exp5_candidate_pool.csv）を出力する。

出力はメタ情報のみ（設問本文は保存・再掲しない＝著作権セーフ）。
最終的なカテゴリ確定と公式PDFとの突合は人手（＋Claudeの判定）で行う。

依存: pip install pdfplumber      （※ pip install --upgrade は使わないこと）
実行: python exp5_build_item_pool.py
特徴: ダウンロード済みPDFはスキップ（レジューム）、進捗表示つき。
"""

import re, sys, time, csv, pathlib, urllib.request, traceback

# ---- 対象ページ（保健師=03 / 助産師=04 / 看護師=05。看護師の05のみ使用）----
YEAR_PAGES = {
    "115_2026": "https://www.mhlw.go.jp/seisakunitsuite/bunya/kenkou_iryou/iryou/topics/tp260424-03_04_05.html",
    "114_2025": "https://www.mhlw.go.jp/seisakunitsuite/bunya/kenkou_iryou/iryou/topics/tp250428-03_04_05.html",
    "113_2024": "https://www.mhlw.go.jp/seisakunitsuite/bunya/kenkou_iryou/iryou/topics/tp240424-03_04_05.html",
    "112_2023": "https://www.mhlw.go.jp/seisakunitsuite/bunya/kenkou_iryou/iryou/topics/tp230524-03_04_05.html",
    "111_2022": "https://www.mhlw.go.jp/seisakunitsuite/bunya/kenkou_iryou/iryou/topics/tp220421-03_04_05.html",
    "110_2021": "https://www.mhlw.go.jp/seisakunitsuite/bunya/kenkou_iryou/iryou/topics/tp210416-03_04_05.html",
    "109_2020": "https://www.mhlw.go.jp/seisakunitsuite/bunya/kenkou_iryou/iryou/topics/tp200414-03_04_05.html",
    "108_2019": "https://www.mhlw.go.jp/seisakunitsuite/bunya/kenkou_iryou/iryou/topics/tp190415-03_04_05.html",
    "107_2018": "https://www.mhlw.go.jp/seisakunitsuite/bunya/kenkou_iryou/iryou/topics/tp180511-03_04_05.html",
    "106_2017": "https://www.mhlw.go.jp/seisakunitsuite/bunya/kenkou_iryou/iryou/topics/tp170425-03_04_05.html",
}
# 直近5年(>=2022)=学習データ汚染が相対的に小さい。古い年ほど汚染↑（解析で層別する）。

OUT = pathlib.Path("exp5_pool"); OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research; nursing-LLM-consistency)"}

# 倫理・優先のキーワード（カテゴリ「候補」抽出用。最終判定は人手/Claude）
ETHICS = ["倫理","守秘","秘密","個人情報","プライバシー","インフォームド","同意",
          "自己決定","意思決定","意思","権利","アドボカ","尊厳","身体拘束","抑制",
          "虐待","終末期","看取り","アドバンス","DNAR","代理","家族への説明","告知"]
PRIORITY = ["優先","最も優先","まず","最初に","初めに","緊急度","トリアージ",
            "直ちに","早急","最初","最優先"]
IMAGE = ["別冊","図","写真","心電図","心音","波形","エックス線","Ｘ線","X線","X-",
         "CT","MRI","超音波","エコー","画像","グラフ","表を","スケール","眼底"]

def fetch(url, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def _select_05(html):
    """ページHTMLから看護師(05)の本試験PDFを選ぶ。
       午前=05a_01, 午後=05c_01, 正答=05seitou。再試(05b/05d)・別冊(_02)は使わない。"""
    def absu(h): return h if h.startswith("http") else "https://www.mhlw.go.jp" + h
    all05 = [absu(h) for h in re.findall(r'href="([^"]*-05[^"]*\.pdf)"', html)]
    def pick(pat): return next((u for u in all05 if re.search(pat, u)), None)
    am  = pick(r'-05a_01\.pdf$')    # 本試験 午前
    pm  = pick(r'-05c_01\.pdf$')    # 本試験 午後
    key = pick(r'-05seitou\.pdf$')  # 本試験 正答
    # 保険: 05a_01/05c_01 が無い年はラベルから本試験(再試・別冊以外)を拾う
    if am is None or pm is None:
        pairs = re.findall(
            r'<td[^>]*>([\s\S]*?)</td>\s*<td[^>]*>\s*<a[^>]*href="([^"]*-05[^"]*\.pdf)"', html)
        for label, href in pairs:
            lab = re.sub(r'<[^>]+>', '', label); u = absu(href)
            if ("再" in lab) or ("別冊" in lab) or ("_02" in u):
                continue
            if am is None and "午前" in lab:
                am = u
            if pm is None and "午後" in lab:
                pm = u
    return am, pm, key

def find_nursing_pdfs(page_url):
    """看護師(05)の本試験 午前/午後/正答 PDF URL を返す。"""
    raw = fetch(page_url)
    html = raw.decode("cp932", "replace")
    return _select_05(html)

def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    skip (exists): {dest.name}"); return dest
    print(f"    downloading: {url}")
    data = fetch(url)
    dest.write_bytes(data)
    print(f"      -> {dest.name} ({len(data)//1024} KB)")
    return dest

def extract_text(pdf_path):
    import pdfplumber, unicodedata
    out = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, pg in enumerate(pdf.pages):
            out.append(pg.extract_text() or "")
    # NFKC: 全角数字/記号→半角（１．→1. , ＡＭ→AM, 全角空白→空白）で書式差を吸収
    return unicodedata.normalize("NFKC", "\n".join(out))

def parse_answer_key(txt):
    """正答PDFテキスト -> {'AM1':'3', 'PM120':'4', ...}。複数選択は連結('24','135')。
    対応書式: (a) 'AM1 3'/'PM120 4'（113-114）
              (b) '午前 1 3'/'午後 …'（保険）
              (c) 'A001 3'/'B045 2'（A=午前, B=午後。106-112,115）。"""
    keys = {}
    # (a) AM/PM 形式
    for m in re.finditer(r'\b(AM|PM)\s*([0-9]{1,3})\s+([0-9]{1,3})\b', txt):
        keys.setdefault(f"{m.group(1)}{int(m.group(2))}", m.group(3))
    # (b) 午前/午後 形式
    for jp, en in (("午前", "AM"), ("午後", "PM")):
        for m in re.finditer(jp + r'\s*([0-9]{1,3})\s+([0-9]{1,3})', txt):
            keys.setdefault(f"{en}{int(m.group(1))}", m.group(2))
    # (c) A###/B### 形式: ヘッダ/表題（『正答』を含む行）を除去し、設問番号トークンを走査。
    #     次の設問番号までの数字を連結（'4 5'→'45'＝複数選択, 空→採点除外）。
    clean = "\n".join(L for L in txt.split("\n") if "正答" not in L)
    toks = list(re.finditer(r'([AB])([0-9]{3})', clean))
    for i, mt in enumerate(toks):
        sess = "AM" if mt.group(1) == "A" else "PM"
        num = int(mt.group(2))
        nxt = toks[i + 1].start() if i + 1 < len(toks) else len(clean)
        ans = re.sub(r'\D', '', clean[mt.end():nxt])
        if ans:
            keys.setdefault(f"{sess}{num}", ans)
    return keys

def segment_items(txt):
    """本文テキストを設問番号で分割（best-effort）。例題(201-204)以降のみ。
       返り値: {qnum:int -> segment:str}"""
    # 例題ブロックの後ろから開始（『1 …』が最初の本問）
    start = 0
    m_ex = re.search(r'(DKIX|DLIX|D[A-Z]{3})-05', txt)  # 例題後のページ識別子近辺
    anchors = [(mm.start(), int(mm.group(1)))
               for mm in re.finditer(r'(?m)^\s*([0-9]{1,3})[ \u3000]', txt)]
    # 単調増加（1,2,3…）になるアンカーだけ残す（誤検出除去）
    seq = []
    expected = 1
    for pos, n in anchors:
        if n == expected:
            seq.append((pos, n)); expected += 1
        elif n == expected + 1 and seq:   # 1問パース漏れを許容
            seq.append((pos, n)); expected = n + 1
    items = {}
    for i, (pos, n) in enumerate(seq):
        end = seq[i+1][0] if i+1 < len(seq) else len(txt)
        items[n] = txt[pos:end]
    return items

def option_count(seg):
    nums = set(int(x) for x in re.findall(r'(?m)(?:^|\s)([1-5])[.)]', seg))
    return max(nums) if nums else 0

def has_any(seg, words):
    return any(w in seg for w in words)

def category_guess(seg):
    if has_any(seg, ETHICS):   return "Ethical?"
    if has_any(seg, PRIORITY): return "Priority?"
    return "Knowledge?"

def main():
    rows = []
    for year, page in YEAR_PAGES.items():
        print(f"\n=== {year} ===")
        try:
            am, pm, key = find_nursing_pdfs(page)
        except Exception as e:
            print(f"  [WARN] page parse failed: {e}"); continue
        print(f"  午前本文: {am}\n  午後本文: {pm}\n  正答    : {key}")
        ydir = OUT / year; ydir.mkdir(exist_ok=True)
        try:
            key_txt = extract_text(download(key, ydir/"seitou.pdf")) if key else ""
        except Exception as e:
            print(f"  [WARN] key extract failed: {e}"); key_txt = ""
        keymap = parse_answer_key(key_txt)
        for sess, url in (("AM", am), ("PM", pm)):
            if not url:
                print(f"  [WARN] {sess} url missing"); continue
            try:
                txt = extract_text(download(url, ydir / f"{sess}_main.pdf"))
            except Exception as e:
                print(f"  [WARN] {sess} extract failed: {e}"); continue
            items = segment_items(txt)
            print(f"  {sess}: parsed {len(items)} item segments")
            for q, seg in sorted(items.items()):
                kid = f"{sess}{q}"
                official = keymap.get(kid, "")
                single = (official.isdigit() and len(official) == 1)
                opt = option_count(seg)
                img = has_any(seg, IMAGE)
                cat = category_guess(seg)
                key_letter = {"1":"A","2":"B","3":"C","4":"D","5":"E"}.get(official, "") if single else ""
                # 採用は「単一解（公式正答が1桁）」かつ「非画像依存」で判定。
                # 選択肢数(k=4/5)は記録のみ（後で公式PDFと突合し、NCIは項目ごとkで算出）。
                include = bool(single and (not img))
                rows.append(dict(
                    item_uid=f"{year}-{sess}-{q}", year=year,
                    recent=("Y" if int(year.split("_")[1]) >= 2022 else "N"),
                    session=sess, q_number=q,
                    option_count=opt, single_best=("Y" if single else "N"),
                    official_key_num=official, official_key_letter=key_letter,
                    image_dependent=("Y" if img else "N"),
                    category_guess=cat, include_candidate=("Y" if include else "N"),
                    notes=""))
            time.sleep(1)
    # 出力
    csv_path = OUT / "exp5_candidate_pool.csv"
    cols = ["item_uid","year","recent","session","q_number","option_count","single_best",
            "official_key_num","official_key_letter","image_dependent",
            "category_guess","include_candidate","notes"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    # 集計
    from collections import Counter
    inc = [r for r in rows if r["include_candidate"] == "Y"]
    cat_total = Counter(r["category_guess"] for r in inc)
    cat_recent = Counter(r["category_guess"] for r in inc if r["recent"] == "Y")
    by = Counter((r["year"], r["category_guess"]) for r in inc)
    print(f"\n===== SUMMARY =====\n候補(include_candidate=Y) 合計: {len(inc)} / 全{len(rows)}")
    print("--- カテゴリ別合計（括弧内=直近5年/低汚染分）---")
    for c in ("Knowledge?", "Priority?", "Ethical?"):
        print(f"  {c:12s} {cat_total.get(c,0):4d}  (recent {cat_recent.get(c,0)})")
    print("--- 年×カテゴリ ---")
    for (y, c), n in sorted(by.items()):
        print(f"  {y:10s} {c:12s} {n}")
    print("--- 年別診断（どの段階で落ちたか）: rows / 正答取得 / 単一解 / 画像 / 採用 ---")
    for y in sorted(set(r["year"] for r in rows)):
        rr = [r for r in rows if r["year"] == y]
        keyed = sum(1 for r in rr if r["official_key_num"])
        sb = sum(1 for r in rr if r["single_best"] == "Y")
        im = sum(1 for r in rr if r["image_dependent"] == "Y")
        ic = sum(1 for r in rr if r["include_candidate"] == "Y")
        print(f"  {y:10s} rows {len(rr):4d}  keyed {keyed:4d}  single {sb:4d}  image {im:4d}  incl {ic:4d}")
    print(f"\n出力: {csv_path.resolve()}")
    print("次の手順: この候補CSVをClaudeに貼って、Ethical?/Priority? の最終確定と")
    print("各カテゴリ40問（最低30）への絞り込みを行う。公式正答・選択肢は必ず公式PDFで突合する。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断（再実行で続行：DL済みPDFはスキップ）")
    except Exception:
        traceback.print_exc()
        print("\n[ERROR] 失敗。pip install pdfplumber を確認。--upgrade は使わないこと。")
