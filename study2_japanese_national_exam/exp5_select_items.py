#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp5_select_items.py  —  実験5 項目選抜（カテゴリ最終確定の支援）

exp5_build_item_pool.py 実行後に使う。やること:
  (1) 各候補設問の「抜粋(snippet, 80字)」を本文(AM_main/PM_main)から生成
  (2) 倫理/優先を精緻化キーワードで再判定（一次判定より高精度）
  (3) 直近5年(低汚染)を優先して各カテゴリ最大 TARGET 問を自動選抜
  (4) 検証用CSVを出力:
        - exp5_review.csv          … 倫理/優先の全候補(抜粋つき) ← これをClaudeに貼る
        - exp5_knowledge_top.csv   … 知識の上位候補(抜粋つき, 直近優先)
        - exp5_selected_auto.csv   … 自動選抜の暫定120

依存: pip install pdfplumber     （※ --upgrade は付けない）
実行: python exp5_select_items.py
"""
import re, csv, pathlib, unicodedata

POOL = pathlib.Path("exp5_pool")
CAND = POOL / "exp5_candidate_pool.csv"
TARGET = 30          # 各カテゴリ目標 = 30問（計90問設計）。120問に戻すなら 40 に変更。
SNIPPET_LEN = 80     # 抜粋の文字数（分類判定用の最小限）

# --- 精緻化キーワード（一次判定の広いキーワードより厳密）---
STRONG_ETHICAL = [
    "倫理", "守秘", "秘密保持", "個人情報", "プライバシー", "インフォームド",
    "説明と同意", "自己決定", "意思決定支援", "意思決定を支援", "アドバンス",
    "尊厳", "権利擁護", "アドボカ", "身体拘束", "抑制帯", "虐待", "終末期",
    "看取り", "ターミナル", "DNAR", "DNR", "リビングウィル", "代理意思",
    "代理決定", "告知", "延命", "エンドオブライフ", "倫理的",
]
STRONG_PRIORITY = [
    "最も優先", "優先順位", "最優先", "まず行う", "まず実施", "まず確認",
    "まず観察", "最初に行う", "最初に実施", "最初に確認", "直ちに", "ただちに",
    "トリアージ", "緊急度", "救命", "最も適切な対応", "最も適切な看護",
    "最初に", "まず",
]

def z(s):
    return unicodedata.normalize("NFKC", s or "")

def extract_pages(pdf_path):
    import pdfplumber
    out = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pg in pdf.pages:
            out.append(pg.extract_text() or "")
    return z("\n".join(out))

def segment(txt):
    anchors = [(m.start(), int(m.group(1)))
               for m in re.finditer(r'(?m)^\s*([0-9]{1,3})[ \u3000]', txt)]
    seq = []; exp = 1
    for pos, n in anchors:
        if n == exp:
            seq.append((pos, n)); exp += 1
        elif n == exp + 1 and seq:
            seq.append((pos, n)); exp = n + 1
    items = {}
    for i, (pos, n) in enumerate(seq):
        end = seq[i + 1][0] if i + 1 < len(seq) else len(txt)
        items[n] = txt[pos:end]
    return items

def snippet(seg, n=SNIPPET_LEN):
    s = re.sub(r'^\s*[0-9]{1,3}[ \u3000]', '', seg)   # 設問番号を除去
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:n]

def refined_cat(seg):
    if any(w in seg for w in STRONG_ETHICAL):
        return "Ethical"
    if any(w in seg for w in STRONG_PRIORITY):
        return "Priority"
    return "Knowledge"

def main():
    if not CAND.exists():
        raise SystemExit("[NG] exp5_pool/exp5_candidate_pool.csv が見つかりません。"
                         "先に exp5_build_item_pool.py を実行してください。")
    rows = [r for r in csv.DictReader(open(CAND, encoding="utf-8-sig"))
            if r["include_candidate"] == "Y"]

    # 年ごとに本文を展開して stem を取得
    stems = {}
    years = sorted(set(r["year"] for r in rows))
    for y in years:
        for sess, fn in (("AM", "AM_main.pdf"), ("PM", "PM_main.pdf")):
            p = POOL / y / fn
            if not p.exists():
                print(f"  [WARN] {p} なし（スキップ）"); continue
            for q, seg in segment(extract_pages(p)).items():
                stems[(y, sess, int(q))] = seg
        print(f"  抽出済: {y}")

    # 抜粋＋再分類
    items = []
    for r in rows:
        key = (r["year"], r["session"], int(r["q_number"]))
        seg = stems.get(key, "")
        items.append(dict(
            item_uid=r["item_uid"], year=r["year"], recent=r["recent"],
            session=r["session"], q_number=r["q_number"],
            option_count=r["option_count"], official_key_letter=r["official_key_letter"],
            first_pass=r["category_guess"].rstrip("?"),
            refined_category=(refined_cat(seg) if seg else "Knowledge"),
            snippet=snippet(seg)))

    def sortkey(it):
        return (0 if it["recent"] == "Y" else 1,
                -int(it["year"].split("_")[1]), it["session"], int(it["q_number"]))

    sel = {}
    for cat in ("Ethical", "Priority", "Knowledge"):
        pool = sorted([it for it in items if it["refined_category"] == cat], key=sortkey)
        sel[cat] = pool[:TARGET]

    cols = ["item_uid", "year", "recent", "session", "q_number", "option_count",
            "official_key_letter", "first_pass", "refined_category", "snippet"]

    # 出力1: 自動選抜（暫定120）
    with open(POOL / "exp5_selected_auto.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for cat in ("Knowledge", "Priority", "Ethical"):
            for it in sel[cat]:
                w.writerow(it)

    # 出力2: 倫理/優先の「全候補」(抜粋つき, selectedフラグ) ← Claudeが最終確定する
    with open(POOL / "exp5_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["selected"] + cols); w.writeheader()
        for cat in ("Ethical", "Priority"):
            # 再判定 cat か、または一次判定が同カテゴリ のものを全て候補に出す（取りこぼし防止）
            allcat = [it for it in items
                      if it["refined_category"] == cat or it["first_pass"] == cat]
            allcat = sorted(allcat, key=sortkey)
            chosen = {x["item_uid"] for x in sel[cat]}
            for it in allcat:
                w.writerow(dict(selected=("Y" if it["item_uid"] in chosen else ""), **it))

    # 出力3: 知識 上位（直近優先, 抜粋つき, スポット確認用）
    with open(POOL / "exp5_knowledge_top.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for it in sel["Knowledge"]:
            w.writerow(it)

    # サマリ
    print("\n===== 選抜サマリ（精緻化分類）=====")
    for cat in ("Knowledge", "Priority", "Ethical"):
        alln = sum(1 for it in items if it["refined_category"] == cat)
        rec = sum(1 for it in sel[cat] if it["recent"] == "Y")
        print(f"  {cat:10s} 選抜 {len(sel[cat]):3d} / 再判定候補 {alln:4d}  (選抜中の直近5年 {rec})")
    ep = sum(1 for it in items if it["refined_category"] in ("Ethical", "Priority")
             or it["first_pass"] in ("Ethical", "Priority"))
    print(f"  検証用 exp5_review.csv 行数(倫理+優先 全候補): 約 {ep}")
    print("出力: exp5_review.csv / exp5_knowledge_top.csv / exp5_selected_auto.csv")
    print("次: exp5_review.csv の中身をClaudeに貼ってください → 一問ずつ分類を最終確定します。")

if __name__ == "__main__":
    main()
