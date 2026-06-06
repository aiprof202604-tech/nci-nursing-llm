#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp5_run_full.py  —  実験5 本番v2（PREVIEW_ONLY=False / TRIALS=30）
  追加: 進捗にエラー数を併記／課金切れ・認証エラーを検知したら即停止し原因を表示。
  90問×3モデル×各温度×30試行=29,700コール。resume・並列8・進捗100。
  キーは apikeys.txt から読込（set不要）。スモークraw.csvは残すこと（trial1流用）。

入力（すべて C:\\Users\\htajima\\Desktop\\HIR 配下）:
  - exp5_final_ep.csv            … 倫理・法規30＋優先30（Claudeが確定）
  - exp5_pool/exp5_knowledge_top.csv … 知識30（選抜スクリプトが出力）
  - exp5_pool/<year>/AM_main.pdf, PM_main.pdf … 本試験本文（取得済み）

処理:
  1) PyMuPDF(fitz)で本文を高精度抽出（pdfplumberのcid破損を回避）
  2) 各設問の「問題文」と「選択肢1..k」を分離し、(cid:)等の破損を検証
  3) PREVIEW: exp5_pool/exp5_prompts.csv に全プロンプトを書き出し、破損件数を報告
     → ここで一旦止めて、人/Claudeが問題文の健全性を確認する
  4) RUN: 3モデル × 各温度 × 30試行 を resume/並列/進捗つきで実行
     結果は exp5_results/raw.csv に逐次保存（NCIは別スクリプトで算出）

依存: pip install pymupdf openai anthropic google-genai
      （※ google-genai が新・推奨SDK。無ければ google-generativeai にフォールバック）
      （※ pip install --upgrade は使わない）
APIキー（環境変数。cmdなら set, PowerShellなら $env:）:
  OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY
実行: python exp5_run_queries.py
"""
import os, re, csv, sys, json, time, pathlib, unicodedata, threading, collections
from concurrent.futures import ThreadPoolExecutor, as_completed

def classify_error(msg):
    """API例外メッセージを 致命的(fatal)＝課金切れ/認証 か 一時的(transient)＝レート制限等 に分類。
       fatal は再試行しても無駄なので実行を停止する。transient はリトライ対象。"""
    m = (msg or "").lower()
    fatal = [
        "credit balance is too low",        # Anthropic 残高不足
        "insufficient_quota", "exceeded your current quota",  # OpenAI クレジット切れ
        "billing", "payment",               # 課金一般
        "api key not valid", "api_key_invalid", "api key expired",
        "invalid x-api-key", "invalid_api_key",
        "authentication", "unauthenticated", "permission denied", "permission_denied",
        "error code: 401", "error code: 403", "status 401", "status 403",
    ]
    return "fatal" if any(k in m for k in fatal) else "transient"

# ============================ 設定 ============================
PREVIEW_ONLY = False  # ★本番v2（TRIALS=30 / エラー可視化・致命的エラー即停止）
TRIALS   = 30
WORKERS  = 8          # 並列数（レート制限が出たら下げる）

# ★前回の実験で使った“正確なモデル名”に合わせて確認してください ★
MODELS = [
    {"name": "gpt-4o",                "provider": "openai",    "model": "gpt-4o",
     "temps": [0.0, 0.5, 1.0, 1.5]},
    {"name": "claude-opus-4.5",       "provider": "anthropic", "model": "claude-opus-4-5",
     "temps": [0.0, 0.5, 1.0]},        # Claudeは temperature<=1.0
    {"name": "gemini-2.5-flash-lite", "provider": "google",    "model": "gemini-2.5-flash-lite",
     "temps": [0.0, 0.5, 1.0, 1.5]},
]

HIR   = pathlib.Path(".")
POOL  = HIR / "exp5_pool"
RES   = HIR / "exp5_results"; RES.mkdir(exist_ok=True)
RAW   = RES / "raw.csv"
PROMPTS = POOL / "exp5_prompts.csv"

# システム/指示文（日本語ネイティブ出題）
INSTRUCTION = ("次の問題に対して、最も適切な選択肢を1つだけ選び、その番号だけを半角数字で答えてください。"
               "番号以外の文字は一切出力しないでください。")

# ============================ 抽出・整形 ============================
def z(s):
    return unicodedata.normalize("NFKC", s or "")

def extract_text_fitz(pdf_path):
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    # sort=True: 位置順に整列して2段組の読み順問題を回避（全年で設問番号を検出）
    parts = [pg.get_text("text", sort=True) for pg in doc]
    doc.close()
    return z("\n".join(parts))

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

def parse_stem_options(seg):
    """設問セグメント -> (stem, [opt...])。選択肢 1.,2.,... を自動検出（4 or 5択想定）。
       検出できなければ (None, None)。CSVのoption_countには依存しない。"""
    seg = re.sub(r'^\s*[0-9]{1,3}[ \u3000]', '', seg.strip())  # 先頭の設問番号を除去
    marks = []
    for m in re.finditer(r'(?:(?<=\s)|^)([1-5])[.\．、]\s*', seg):
        marks.append((int(m.group(1)), m.start(), m.end()))
    # 1,2,3,... と昇順連続する分だけ採用（最大5）
    chosen = []
    expect = 1
    for num, s, e in marks:
        if num == expect:
            chosen.append((num, s, e)); expect += 1
            if expect > 5:
                break
    k = len(chosen)
    if k < 4:                      # 看護MCQは4/5択。未満は解析失敗として除外。
        return None, None
    stem = seg[:chosen[0][1]].strip()
    opts = []
    for i in range(k):
        s_e = chosen[i][2]
        nxt = chosen[i + 1][1] if i + 1 < k else len(seg)
        opts.append(re.sub(r'\s+', ' ', seg[s_e:nxt]).strip())
    stem = re.sub(r'\s+', ' ', stem).strip()
    if not stem or any(not o for o in opts):
        return None, None
    return stem, opts

def looks_broken(*texts):
    blob = " ".join(t for t in texts if t)
    if "(cid:" in blob:                 # フォント未対応の取りこぼし
        return True
    if "\ufffd" in blob:                # 置換文字
        return True
    if re.search(r'[\uF000-\uF8FF]', blob):  # 私用領域（外字化け）
        return True
    return False

def context_ok(stem):
    """前問の症例を前提とする『文脈欠落』サブ設問を除外する。
       特定の人物参照(Aさん/母親/裸の児 等)があるのに、年齢/状況記述が無く短い場合は不採用。"""
    s = stem.strip()
    # 文頭の指示語（このとき/その後/これら 等）→ 前問依存とみなす（文中の「その内容」等は除外しない）
    if re.match(r'^(この|その|これ|それ|同[じ表患児])', s):
        return False
    # 本文内に症例記述があるとみなせる手掛かり
    has_ctx = bool(re.search(r'\d+\s*歳', s)) or "である" in s or len(s) >= 75
    # 複合語(新生児/乳児/児童/胎児等)ではない『裸の児＋助詞』＝前問の登場児
    if re.search(r'(?<![\u4e00-\u9fff])児[へにはがをのも、]', s) and not has_ctx:
        return False
    refs = ("Aさん", "A さん", "A君", "A 君", "Aちゃん", "A ちゃん",
            "B さん", "Bちゃん", "患児", "母親", "父親", "息子")
    if any(x in s for x in refs) and not has_ctx:
        return False
    return True

def build_prompt(stem, opts):
    lines = [INSTRUCTION, "", stem]
    for i, o in enumerate(opts, 1):
        lines.append(f"{i}. {o}")
    lines.append("")
    lines.append("答え：")
    return "\n".join(lines)

# ============================ 項目ロード ============================
# === Claudeの精査による除外（純粋な解剖・薬理・統計・制度トリビア等を倫理/優先から除外）===
ETH_EXCLUDE = {
 "114_2025-AM-18","114_2025-AM-57","113_2024-AM-31","113_2024-AM-77","113_2024-AM-97",
 "111_2022-AM-59","111_2022-AM-63","108_2019-AM-25","107_2018-PM-27","107_2018-PM-68",
 "107_2018-PM-79","106_2017-AM-15","106_2017-AM-58","106_2017-PM-43","106_2017-AM-80",
 "106_2017-PM-55",
}
PRI_EXCLUDE = {
 "115_2026-AM-42","115_2026-PM-10","115_2026-PM-34","114_2025-PM-18","114_2025-AM-40",
 "114_2025-AM-41","113_2024-AM-50","113_2024-AM-57","111_2022-AM-24","111_2022-AM-27",
 "107_2018-AM-33","107_2018-AM-36","107_2018-PM-76","106_2017-AM-25","106_2017-AM-49",
 "106_2017-PM-24",
}
TARGET_PER_CAT = 30   # 各カテゴリ目標（90問設計）

def _pools_from_candidates():
    cand = POOL / "exp5_candidate_pool.csv"
    if not cand.exists():
        sys.exit("[NG] exp5_pool/exp5_candidate_pool.csv が見つかりません（exp5_build_item_pool.py を実行）。")
    rows = [r for r in csv.DictReader(open(cand, encoding="utf-8-sig")) if r["include_candidate"] == "Y"]
    def sk(r):
        return (0 if r["recent"] == "Y" else 1, -int(r["year"].split("_")[1]), r["session"], int(r["q_number"]))
    eth = sorted([r for r in rows if r["category_guess"] == "Ethical?"  and r["item_uid"] not in ETH_EXCLUDE], key=sk)
    pri = sorted([r for r in rows if r["category_guess"] == "Priority?" and r["item_uid"] not in PRI_EXCLUDE], key=sk)
    kno = sorted([r for r in rows if r["category_guess"] == "Knowledge?"], key=sk)
    return [("EthicalLegal", eth), ("Priority", pri), ("Knowledge", kno)]

def load_items():
    """候補プールから、本文抽出が健全な項目を各カテゴリ TARGET_PER_CAT 件ずつ直近優先で選抜。
       抽出失敗（セグメント無し/選択肢不検出/文字化け）は自動的に次候補へ置換する。"""
    pools = _pools_from_candidates()
    cache = {}
    def stems_for(year, sess):
        if (year, sess) not in cache:
            fn = POOL / year / f"{sess}_main.pdf"
            cache[(year, sess)] = segment(extract_text_fitz(fn)) if fn.exists() else {}
        return cache[(year, sess)]

    items = []; report = []
    for cat, pool in pools:
        got = []; skip = {"seg": 0, "opt": 0, "broken": 0, "context": 0}
        for r in pool:
            if len(got) >= TARGET_PER_CAT:
                break
            uid = r["item_uid"]
            m = re.match(r'(\d+_\d+)-(AM|PM)-(\d+)', uid)
            if not m:
                continue
            year, sess, q = m.group(1), m.group(2), int(m.group(3))
            seg = stems_for(year, sess).get(q, "")
            if not seg:
                skip["seg"] += 1; continue
            try:
                stem, opts = parse_stem_options(seg)
            except Exception:
                skip["opt"] += 1; continue
            if stem is None:
                skip["opt"] += 1; continue
            if looks_broken(stem, *opts):
                skip["broken"] += 1; continue
            if not context_ok(stem):
                skip["context"] += 1; continue
            got.append(dict(item_uid=uid, category=cat, k=len(opts),
                            key_letter=r["official_key_letter"], recent=r["recent"],
                            stem=stem, options=opts, prompt=build_prompt(stem, opts)))
        items += got
        report.append((cat, len(got), len(pool), skip))
    return items, report

# ============================ API 呼び出し ============================
_openai = _anthropic = None
_gclient = None        # Geminiクライアント
_gmode = None          # ("new", types) or ("old", None)
def _lazy_clients():
    global _openai, _anthropic, _gclient, _gmode
    providers = {m["provider"] for m in MODELS}
    if "openai" in providers and _openai is None:
        from openai import OpenAI; _openai = OpenAI()
    if "anthropic" in providers and _anthropic is None:
        import anthropic; _anthropic = anthropic.Anthropic()
    if "google" in providers and _gclient is None:
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY（または GEMINI_API_KEY）が未設定です。")
        try:
            # 新・統合SDK（推奨）: pip install google-genai
            from google import genai as g_new
            from google.genai import types as g_types
            _gclient = g_new.Client(api_key=key)
            _gmode = ("new", g_types)
            print("  [Gemini] 新SDK google-genai を使用（thinking無効化）")
        except Exception:
            # 旧SDKフォールバック: pip install google-generativeai
            import google.generativeai as g_old
            g_old.configure(api_key=key)
            _gclient = g_old
            _gmode = ("old", None)
            print("  [Gemini] 旧SDK google-generativeai を使用（※2.5系はthinkingで空応答の恐れ→google-genai推奨）")

def call_model(spec, prompt, temp):
    p = spec["provider"]
    if p == "openai":
        r = _openai.chat.completions.create(
            model=spec["model"], messages=[{"role": "user", "content": prompt}],
            temperature=temp, max_tokens=8)
        return r.choices[0].message.content or ""
    if p == "anthropic":
        r = _anthropic.messages.create(
            model=spec["model"], max_tokens=8, temperature=temp,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    if p == "google":
        mode, gtypes = _gmode
        if mode == "new":
            cfg = gtypes.GenerateContentConfig(
                temperature=temp, max_output_tokens=16,
                thinking_config=gtypes.ThinkingConfig(thinking_budget=0))  # 思考無効＝出力を答えに使う
            r = _gclient.models.generate_content(model=spec["model"], contents=prompt, config=cfg)
        else:
            gm = _gclient.GenerativeModel(spec["model"])
            r = gm.generate_content(prompt, generation_config={"temperature": temp, "max_output_tokens": 16})
        # 安全フィルタ等で .text が例外を投げる版があるため堅牢に取り出す
        try:
            t = r.text
            if t:
                return t
        except Exception:
            pass
        try:
            return "".join(getattr(pt, "text", "") for pt in r.candidates[0].content.parts) or ""
        except Exception:
            return ""
    raise ValueError(p)

NUM2LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
def parse_choice(text):
    m = re.search(r'[1-5]', text or "")
    return NUM2LETTER.get(m.group(0)) if m else ""

# ============================ 実行 ============================
def load_done():
    """成功した応答だけを『完了』とみなす。__ERROR__行は未完了として再試行対象に残す。"""
    done = set()
    if RAW.exists():
        for r in csv.DictReader(open(RAW, encoding="utf-8-sig")):
            if (r.get("raw_response") or "").startswith("__ERROR__"):
                continue   # エラー行はスキップ＝次回再試行する
            done.add((r["item_uid"], r["model"], r["temp"], r["trial"]))
    return done

def _load_key_file():
    """HIRフォルダの apikeys.txt があれば、そこからAPIキーを読み、環境変数より優先する。
       形式: 各行  NAME=値   （# 始まりの行・空行は無視。値の前後の引用符は除去）
       これで set / setx の優先順位やウィンドウ依存の問題を回避する。"""
    f = HIR / "apikeys.txt"
    if not f.exists():
        return
    loaded = []
    for line in f.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if v:
            os.environ[k] = v          # ファイルの値で環境変数を上書き
            loaded.append(k)
    if loaded:
        print(f"  [keys] apikeys.txt を読み込み、{', '.join(loaded)} を設定（環境変数より優先）")

def main():
    _load_key_file()
    items, report = load_items()
    print("=== 選抜結果（各カテゴリ 健全な30問を直近優先で自動選抜）===")
    for cat, n, pool, skip in report:
        print(f"  {cat:13s} 選抜 {n:2d}/{TARGET_PER_CAT}  候補{pool:4d}  "
              f"(除外: セグ無{skip['seg']} 選択肢{skip['opt']} 文字化け{skip['broken']} 文脈欠落{skip['context']})")
    print(f"  合計 健全 {len(items)} 問")

    # プロンプト プレビュー出力
    with open(PROMPTS, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["item_uid", "category", "recent", "k", "key_letter", "stem", "options", "prompt"])
        for it in items:
            w.writerow([it["item_uid"], it["category"], it.get("recent", ""), it["k"], it["key_letter"],
                        it["stem"], " | ".join(it["options"]), it["prompt"]])
    print(f"\nプロンプトを書き出しました: {PROMPTS.resolve()}")
    cats = {}
    for it in items:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    print("  カテゴリ内訳:", cats)

    if PREVIEW_ONLY:
        print("\n[PREVIEW_ONLY=True] API呼び出しは行いません。")
        print("→ exp5_prompts.csv の問題文を確認し、問題なければ本スクリプト冒頭の PREVIEW_ONLY=False にして再実行してください。")
        return

    # ---- 実行 ----
    _lazy_clients()
    tasks = []
    for it in items:
        for spec in MODELS:
            for t in spec["temps"]:
                for trial in range(1, TRIALS + 1):
                    tasks.append((it, spec, t, trial))
    done = load_done()
    todo = [x for x in tasks
            if (x[0]["item_uid"], x[1]["name"], f'{x[2]}', f'{x[3]}') not in done]
    print(f"\n総タスク {len(tasks)} / 実行済 {len(done)} / 今回 {len(todo)}")

    lock = threading.Lock()
    newfile = not RAW.exists()
    fout = open(RAW, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(fout)
    if newfile:
        writer.writerow(["item_uid", "category", "model", "temp", "trial",
                         "raw_response", "parsed_letter", "key_letter", "correct"])
        fout.flush()

    counter = {"n": 0}
    err_counter = collections.Counter()      # model -> エラー行数
    stop = threading.Event()                  # 致命的エラー検知で全体停止
    fatal_info = {"msg": "", "model": ""}

    def work(task):
        it, spec, t, trial = task
        if stop.is_set():
            return                            # 致命的エラー後は新規処理しない（resumeで続行）
        raw = None
        for attempt in range(4):
            try:
                raw = call_model(spec, it["prompt"], t)
                break
            except Exception as e:
                msg = f"{type(e).__name__}:{e}"
                if classify_error(msg) == "fatal":
                    with lock:
                        if not stop.is_set():
                            print("\n" + "=" * 64, flush=True)
                            print(f"  ★致命的エラー（課金切れ/認証）を検知: モデル = {spec['name']}", flush=True)
                            print(f"    {msg[:200]}", flush=True)
                            print("  → 安全に停止します。原因（クレジット/APIキー）を解消後、", flush=True)
                            print("     python exp5_run_full.py を再実行すれば続きから再開します。", flush=True)
                            print("=" * 64, flush=True)
                            fatal_info["msg"] = msg[:300]; fatal_info["model"] = spec["name"]
                        stop.set()
                    raw = f"__ERROR__:{msg}"
                    break
                if attempt == 3:
                    raw = f"__ERROR__:{msg}"                 # リトライ尽きた一時エラー
                else:
                    time.sleep(2 * (attempt + 1))           # 2,4,6秒バックオフ
        is_err = (raw or "").startswith("__ERROR__")
        letter = "" if is_err else parse_choice(raw)
        correct = int(letter == it["key_letter"]) if letter else 0
        with lock:
            writer.writerow([it["item_uid"], it["category"], spec["name"], t, trial,
                             (raw or "").replace("\n", " ")[:200], letter, it["key_letter"], correct])
            counter["n"] += 1
            if is_err:
                err_counter[spec["name"]] += 1
                if err_counter[spec["name"]] == 1:          # 各モデル初回エラーは即通知
                    print(f"  [注意] {spec['name']} で初のエラー: {(raw or '')[10:130]}", flush=True)
            if counter["n"] % 100 == 0:
                fout.flush()
                te = sum(err_counter.values())
                tail = f" / エラー {te}{dict(err_counter)}" if te else ""
                print(f"  進捗 {counter['n']}/{len(todo)}{tail}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(work, x) for x in todo]
            for _ in as_completed(futs):
                pass
    except KeyboardInterrupt:
        stop.set()
        print("\n中断（再実行で続行：実行済はスキップ）", flush=True)
    finally:
        fout.flush(); fout.close()

    # ---- 終了サマリ ----
    te = sum(err_counter.values())
    done_now = counter["n"]
    print(f"\n完了。処理 {done_now} 件 / 結果: {RAW.resolve()}")
    if te == 0:
        print("  エラー 0 件。全タスク成功。")
    else:
        print(f"  エラー合計 {te} 件: {dict(err_counter)}")
        if fatal_info["msg"]:
            print(f"  ※致命的（要対応）: {fatal_info['model']} — {fatal_info['msg'][:160]}")
            print("  ※クレジット残高/APIキーを確認 → python exp5_run_full.py を再実行（resumeで続き）。")
        else:
            print("  ※一時的エラーの可能性大。再実行すれば該当ぶんだけ自動再試行されます。")
    if stop.is_set() and fatal_info["msg"]:
        print("  ※致命的エラーで途中停止したため、未完了ぶんが残っています（再実行で完了します）。")
    else:
        print("  次: NCI算出スクリプトで セル(item×model×temp)ごとの NCI と正答率を計算します。")

if __name__ == "__main__":
    main()
