# -*- coding: utf-8 -*-
"""
exp5_analyze.py  —  実験5（クロスリンガル再現）のNCI・正答率解析と作図

入力:
  exp5_results/raw.csv         …  ランナー出力（item_uid,category,model,temp,trial,
                                   raw_response,parsed_letter,key_letter,correct）
  exp5_pool/exp5_prompts.csv   …  項目メタ（item_uid,category,recent,k,key_letter,...）

出力（exp5_results/ 配下）:
  exp5_cell_metrics.csv            セル(item×model×temp)別 NCI・正答率・最頻回答
  exp5_summary_model_temp.csv      モデル×温度 平均NCI/正答率（±SD）
  exp5_summary_model_category.csv  モデル×カテゴリ（全温度プール）
  exp5_summary_model_cat_temp.csv  モデル×カテゴリ×温度
  exp5_summary_recent.csv          recent(22-26) vs older 層別
  exp5_shared_blindspots.csv       ベンダー横断の共通盲点（≥2モデルが同一誤答に収束）
  exp5_item_table.csv              項目別 per-model 正答率・NCI（補遺表）
  fig_nci_vs_temp.(png|pdf)        NCI vs 温度
  fig_acc_vs_temp.(png|pdf)        正答率 vs 温度
  fig_nci_by_category.(png|pdf)    モデル×カテゴリ NCI
  fig_nci_vs_acc.(png|pdf)         NCI–正答率の乖離（散布図）

NCI = 1 − H(R)/log2(k)。H(R)は30試行の回答分布のシャノンエントロピー（底2）、
kは当該項目の選択肢数。__ERROR__行は除外。空応答(∅)は1カテゴリとして扱う。
"""
import os, csv, math, statistics as st, pathlib
from collections import defaultdict, Counter

HERE = pathlib.Path(".")
RAW     = HERE / "exp5_results" / "raw.csv"
PROMPTS = HERE / "exp5_pool"    / "exp5_prompts.csv"
OUT     = HERE / "exp5_results"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt-4o", "claude-opus-4.5", "gemini-2.5-flash-lite"]
CATS   = ["Knowledge", "Priority", "EthicalLegal"]
CUD    = {"gpt-4o": "#1F4E79", "claude-opus-4.5": "#C45A0E", "gemini-2.5-flash-lite": "#2E7D54"}
LBL    = {"gpt-4o": "GPT-4o", "claude-opus-4.5": "Claude Opus 4.5", "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite"}

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
def sd(xs):   return st.pstdev(xs) if len(xs) > 1 else 0.0

# ---------- 読み込み ----------
meta = {}
for r in csv.DictReader(open(PROMPTS, encoding="utf-8-sig")):
    meta[r["item_uid"]] = {"k": int(r["k"]), "recent": r["recent"],
                           "cat": r["category"], "key": r["key_letter"]}

raw = [r for r in csv.DictReader(open(RAW, encoding="utf-8-sig"))
       if not (r["raw_response"] or "").startswith("__ERROR__")]

# 同一(item,model,temp,trial)の重複は最後を採用（resume由来の重複対策）
seen = {}
for r in raw:
    seen[(r["item_uid"], r["model"], r["temp"], r["trial"])] = r
raw = list(seen.values())

# ---------- セル別指標 ----------
cells = defaultdict(list)
for r in raw:
    cells[(r["item_uid"], r["model"], r["temp"])].append(r["parsed_letter"] or "∅")

def nci_of(resps, k):
    n = len(resps); c = Counter(resps)
    H = -sum((v/n)*math.log2(v/n) for v in c.values())
    return 1 - H/math.log2(k), H, c.most_common(1)[0][0]

cell_rows = []
for (uid, model, temp), resps in cells.items():
    m = meta[uid]; nci, H, modal = nci_of(resps, m["k"])
    acc = sum(1 for x in resps if x == m["key"]) / len(resps)
    cell_rows.append({"item_uid": uid, "category": m["cat"], "recent": m["recent"],
                      "k": m["k"], "key": m["key"], "model": model, "temp": float(temp),
                      "n_trials": len(resps), "nci": round(nci, 4), "entropy_bits": round(H, 4),
                      "accuracy": round(acc, 4), "modal_answer": modal,
                      "modal_correct": int(modal == m["key"])})

with open(OUT/"exp5_cell_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys())); w.writeheader(); w.writerows(cell_rows)

# ---------- モデル×温度 ----------
mt = defaultdict(lambda: defaultdict(list))
for r in cell_rows: mt[r["model"]][r["temp"]].append((r["nci"], r["accuracy"]))
with open(OUT/"exp5_summary_model_temp.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["model","temp","n_items","mean_NCI","sd_NCI","mean_accuracy","sd_accuracy"])
    for m in MODELS:
        for t in sorted(mt[m]):
            ncis=[a for a,_ in mt[m][t]]; accs=[b for _,b in mt[m][t]]
            w.writerow([m,t,len(ncis),round(mean(ncis),4),round(sd(ncis),4),round(mean(accs),4),round(sd(accs),4)])

# ---------- モデル×カテゴリ（温度プール） ----------
mc = defaultdict(lambda: defaultdict(list))
for r in cell_rows: mc[r["model"]][r["category"]].append((r["nci"], r["accuracy"]))
with open(OUT/"exp5_summary_model_category.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["model","category","n_cells","mean_NCI","sd_NCI","mean_accuracy","sd_accuracy"])
    for m in MODELS:
        for c in CATS:
            ncis=[a for a,_ in mc[m][c]]; accs=[b for _,b in mc[m][c]]
            w.writerow([m,c,len(ncis),round(mean(ncis),4),round(sd(ncis),4),round(mean(accs),4),round(sd(accs),4)])

# ---------- モデル×カテゴリ×温度 ----------
with open(OUT/"exp5_summary_model_cat_temp.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["model","category","temp","n_items","mean_NCI","mean_accuracy"])
    mct = defaultdict(list)
    for r in cell_rows: mct[(r["model"],r["category"],r["temp"])].append((r["nci"],r["accuracy"]))
    for m in MODELS:
        for c in CATS:
            for t in sorted(set(r["temp"] for r in cell_rows if r["model"]==m)):
                v=mct[(m,c,t)]
                if v: w.writerow([m,c,t,len(v),round(mean([a for a,_ in v]),4),round(mean([b for _,b in v]),4)])

# ---------- recent 層別 ----------
with open(OUT/"exp5_summary_recent.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["model","stratum","n_items","mean_NCI","mean_accuracy"])
    for m in MODELS:
        for rc,lab in [("Y","recent_2022_2026"),("N","older_2017_2021")]:
            sub=[r for r in cell_rows if r["model"]==m and r["recent"]==rc]
            ni=len(set(r["item_uid"] for r in sub))
            w.writerow([m,lab,ni,round(mean([r["nci"] for r in sub]),4),round(mean([r["accuracy"] for r in sub]),4)])

# ---------- 優勢回答・共通盲点 ----------
pool = defaultdict(Counter)
for r in raw: pool[(r["item_uid"], r["model"])][r["parsed_letter"] or "∅"] += 1
dom = {}
for (uid,model),c in pool.items():
    tot=sum(c.values()); let,n=c.most_common(1)[0]; dom[(uid,model)]=(let, n/tot)

with open(OUT/"exp5_shared_blindspots.csv", "w", newline="", encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["item_uid","category","key","n_models_wrong_agree","agreed_wrong_answer",
                                 "gpt4o_dom","gpt4o_frac","claude_dom","claude_frac","gemini_dom","gemini_frac","all3_agree_wrong"])
    for uid in sorted(meta, key=lambda u:(meta[u]["cat"],u)):
        key=meta[uid]["key"]; doms={mo:dom[(uid,mo)] for mo in MODELS}
        wc=Counter(v[0] for v in doms.values() if v[0]!=key)
        if wc and wc.most_common(1)[0][1]>=2:
            wrong,nm=wc.most_common(1)[0]
            all3=int(len(set(v[0] for v in doms.values()))==1 and list(doms.values())[0][0]!=key)
            w.writerow([uid,meta[uid]["cat"],key,nm,wrong,
                        doms["gpt-4o"][0],round(doms["gpt-4o"][1],3),
                        doms["claude-opus-4.5"][0],round(doms["claude-opus-4.5"][1],3),
                        doms["gemini-2.5-flash-lite"][0],round(doms["gemini-2.5-flash-lite"][1],3),all3])

# ---------- 項目別 補遺表 ----------
with open(OUT/"exp5_item_table.csv", "w", newline="", encoding="utf-8-sig") as f:
    w=csv.writer(f)
    head=["item_uid","category","recent","k","key"]
    for m in MODELS: head += [f"{m}_NCI", f"{m}_acc"]
    w.writerow(head)
    for uid in sorted(meta, key=lambda u:(meta[u]["cat"],u)):
        row=[uid,meta[uid]["cat"],meta[uid]["recent"],meta[uid]["k"],meta[uid]["key"]]
        for m in MODELS:
            sub=[r for r in cell_rows if r["item_uid"]==uid and r["model"]==m]
            row += [round(mean([r["nci"] for r in sub]),3), round(mean([r["accuracy"] for r in sub]),3)]
        w.writerow(row)

print("[OK] CSV 7種を exp5_results/ に出力しました。")

# ---------- 作図 ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.linewidth": 0.8,
                         "savefig.dpi": 300, "figure.dpi": 150})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUT/f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT/f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Fig1: NCI vs temp
    fig, ax = plt.subplots(figsize=(5.2,4))
    for m in MODELS:
        ts=sorted(mt[m]); ys=[mean([a for a,_ in mt[m][t]]) for t in ts]; es=[sd([a for a,_ in mt[m][t]]) for t in ts]
        ax.errorbar(ts, ys, yerr=es, marker="o", color=CUD[m], label=LBL[m], capsize=3, lw=1.8, ms=6)
    ax.set_xlabel("Sampling temperature"); ax.set_ylabel("Mean NCI"); ax.set_ylim(0.85,1.005)
    ax.set_title("Response consistency (NCI) vs temperature"); ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.3, lw=.5); save(fig, "fig_nci_vs_temp")

    # Fig2: Accuracy vs temp
    fig, ax = plt.subplots(figsize=(5.2,4))
    for m in MODELS:
        ts=sorted(mt[m]); ys=[mean([b for _,b in mt[m][t]]) for t in ts]; es=[sd([b for _,b in mt[m][t]]) for t in ts]
        ax.errorbar(ts, ys, yerr=es, marker="s", color=CUD[m], label=LBL[m], capsize=3, lw=1.8, ms=6)
    ax.set_xlabel("Sampling temperature"); ax.set_ylabel("Mean accuracy"); ax.set_ylim(0.80,1.01)
    ax.set_title("Accuracy vs temperature"); ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.3, lw=.5); save(fig, "fig_acc_vs_temp")

    # Fig3: NCI by category (grouped bars, temp-pooled)
    fig, ax = plt.subplots(figsize=(5.6,4))
    x=range(len(CATS)); width=0.26
    for i,m in enumerate(MODELS):
        ys=[mean([a for a,_ in mc[m][c]]) for c in CATS]
        ax.bar([xx+(i-1)*width for xx in x], ys, width, color=CUD[m], label=LBL[m])
    ax.set_xticks(list(x)); ax.set_xticklabels(["Knowledge","Priority","Ethical/Legal"])
    ax.set_ylabel("Mean NCI"); ax.set_ylim(0.85,1.005); ax.set_title("NCI by category")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.3, lw=.5); save(fig, "fig_nci_by_category")

    # Fig4: NCI vs accuracy scatter (cell-level)
    fig, ax = plt.subplots(figsize=(5,4.4))
    for m in MODELS:
        xs=[r["accuracy"] for r in cell_rows if r["model"]==m]; ys=[r["nci"] for r in cell_rows if r["model"]==m]
        ax.scatter(xs, ys, s=10, alpha=.45, color=CUD[m], label=LBL[m], edgecolors="none")
    ax.set_xlabel("Accuracy (per cell)"); ax.set_ylabel("NCI (per cell)")
    ax.set_title("NCI–accuracy dissociation"); ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.3, lw=.5); save(fig, "fig_nci_vs_acc")

    print("[OK] 図4種（PNG+PDF）を exp5_results/ に出力しました。")
except Exception as e:
    print("[警告] 作図をスキップ（matplotlib未導入か失敗）:", e)
    print("       図が必要なら:  pip install matplotlib")

print("完了。")
