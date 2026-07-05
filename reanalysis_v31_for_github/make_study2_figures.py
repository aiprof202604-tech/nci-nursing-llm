"""Study2中心・再構築版の本文図（3点）を生成。全数値はSTUDY2_source_of_truth.json準拠。"""
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.font_manager as fm
import numpy as np, pandas as pd, json
from matplotlib.ticker import MultipleLocator

for p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
    fm.fontManager.addfont(p)
SANS="Liberation Sans"
mpl.rcParams.update({"font.family":SANS,"font.size":9,"axes.titlesize":9.5,"axes.labelsize":9,
    "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,"axes.linewidth":0.8,
    "axes.edgecolor":"#222","axes.spines.top":False,"axes.spines.right":False,"axes.axisbelow":True,
    "xtick.major.width":0.8,"ytick.major.width":0.8,"xtick.major.size":3,"ytick.major.size":3,
    "lines.solid_capstyle":"round","savefig.dpi":600,"savefig.bbox":"tight","savefig.facecolor":"white",
    "savefig.pad_inches":0.04,"figure.facecolor":"white","pdf.fonttype":42})
COL={"GPT-4o":"#2B5D8A","Claude":"#B8553A","Gemini":"#3E8A6E"}
FILL={"GPT-4o":"#cdddea","Claude":"#f0d3c8","Gemini":"#cce4da"}
MK={"GPT-4o":"o","Claude":"s","Gemini":"^"}
LAB={"GPT-4o":"GPT-4o","Claude":"Claude Opus 4.5","Gemini":"Gemini 2.5 Flash-Lite"}
GRID="#e8e8e8"; REF="#9a9a9a"; ORDER=["GPT-4o","Claude","Gemini"]
OUT="/home/claude/work/IFHSC_figures_v33"; import os; os.makedirs(OUT,exist_ok=True)
SOT=json.load(open("STUDY2_source_of_truth.json"))

def sg(ax,axis="y"): ax.grid(axis=axis,color=GRID,linewidth=0.6,zorder=0); ax.set_axisbelow(True)
def pl(ax,s): ax.text(-0.15,1.04,s,transform=ax.transAxes,fontsize=11,fontweight="bold",va="bottom",ha="left")
def save(fig,n): fig.savefig(f"{OUT}/{n}.png"); fig.savefig(f"{OUT}/{n}.pdf"); plt.close(fig); print(f"  {n}")

# ============ Figure 1: Study2 温度別NCI ============
fig,ax=plt.subplots(figsize=(5.2,3.7)); sg(ax)
ax.axhline(1.0,color=REF,lw=0.7,ls=(0,(4,3)),zorder=1)
for m in ORDER:
    d=SOT["temperature"][m]; T=np.array([float(t) for t in d]); 
    y=np.array([d[t]["NCI"] for t in d]); sd=np.array([d[t]["SD"] for t in d]); n=90
    se=sd/np.sqrt(n)
    ax.fill_between(T,np.clip(y-se,0,1),np.clip(y+se,0,1),color=FILL[m],alpha=0.55,lw=0,zorder=2)
    ax.plot(T,y,color=COL[m],marker=MK[m],ms=4.5,lw=1.6,mfc=COL[m],mec="white",mew=0.5,zorder=4,clip_on=False)
ax.set_xlim(-0.03,1.53); ax.set_ylim(0.92,1.002)
ax.set_xticks([0.0,0.5,1.0,1.5]); ax.xaxis.set_minor_locator(MultipleLocator(0.25)); ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.set_xlabel("Sampling temperature"); ax.set_ylabel("Mean response consistency (1 − normalised entropy)")
h=[plt.Line2D([],[],color=COL[m],marker=MK[m],ms=4.5,lw=1.6,mec="white",mew=0.5,label=LAB[m]) for m in ORDER]
ax.legend(handles=h,loc="lower center",bbox_to_anchor=(0.5,1.01),ncol=3,frameon=False,handletextpad=0.4,columnspacing=1.3,fontsize=7.8)
ax.annotate("Claude: API maximum T = 1.0",xy=(1.0,SOT["temperature"]["Claude"]["1.0"]["NCI"]),xytext=(1.04,0.948),
            fontsize=7,color="#666",va="center",ha="left",arrowprops=dict(arrowstyle="-",color="#aaa",lw=0.7,connectionstyle="arc3,rad=0.2"))
fig.subplots_adjust(top=0.9); save(fig,"Figure1")

# ============ Figure 2: 較正(a) + confidently-wrong残差(b) ============
df=pd.read_csv(str(__import__("pathlib").Path(__file__).resolve().parent / "cells_all.csv")); s2=df[df.study=='S2-JP']
# 較正: Study2のみでreliability bins
def reliab(sub,bins=10):
    edges=np.linspace(0,1,bins+1); xs=[];ys=[]
    for i in range(bins):
        lo,hi=edges[i],edges[i+1]
        mask=(sub.phat>lo)&(sub.phat<=hi) if i>0 else (sub.phat>=lo)&(sub.phat<=hi)
        if mask.sum()==0: continue
        xs.append(sub.phat[mask].mean()); ys.append(sub.modal_correct[mask].mean())
    return xs,ys
fig,(axA,axB)=plt.subplots(1,2,figsize=(7.0,3.3))
sg(axA,"both"); axA.plot([0,1],[0,1],color=REF,lw=0.8,ls=(0,(4,3)),zorder=1,label="Perfect calibration")
for m in ORDER:
    xs,ys=reliab(s2[s2.model==m])
    axA.plot(xs,ys,color=COL[m],marker=MK[m],ms=4.5,lw=1.5,mfc=COL[m],mec="white",mew=0.5,zorder=3,
             label=f"{LAB[m]}  (ECE {SOT['ECE'][m]:.3f})")
axA.set_xlim(0.35,1.02); axA.set_ylim(-0.02,1.03)
axA.xaxis.set_major_locator(MultipleLocator(0.2)); axA.yaxis.set_major_locator(MultipleLocator(0.2))
axA.set_xlabel("Self-consistency confidence\n(modal-answer frequency)"); axA.set_ylabel("Observed accuracy of modal answer")
axA.legend(loc="lower right",frameon=False,fontsize=7,handletextpad=0.4); pl(axA,"a")
# 残差(b): 絶対数併記
sg(axB,"y"); xs=np.arange(len(ORDER))
vals=[SOT["unanimous_but_wrong"][m]["pct"] for m in ORDER]
ns=[SOT["unanimous_but_wrong"][m]["n_items"] for m in ORDER]
axB.bar(xs,vals,width=0.58,color=[COL[m] for m in ORDER],edgecolor="white",linewidth=0.6,zorder=3)
for xi,v,nn in zip(xs,vals,ns):
    axB.text(xi,v+0.12,f"{v:.1f}%\n({nn}/90)",ha="center",va="bottom",fontsize=8,fontweight="bold")
axB.set_xticks(xs); axB.set_xticklabels([LAB[m].replace(" Opus"," \nOpus").replace(" 2.5"," \n2.5") for m in ORDER],fontsize=7.5)
axB.set_ylim(0,max(vals)*1.4); axB.yaxis.set_major_locator(MultipleLocator(2))
axB.set_ylabel("Unanimous (30/30) but wrong (%)"); pl(axB,"b")
fig.subplots_adjust(wspace=0.32,bottom=0.2); save(fig,"Figure2")

# ============ Figure 3: トリアージ（CIつき、Study2のみ90項目）============
np.random.seed(20260703)
def triage_curve_auc(sub,grid):
    s=sub.sort_values("NCI").reset_index(drop=True); n=len(s); w=int((s.modal_correct==0).sum())
    if w==0: return None,None
    cap=np.cumsum((s.modal_correct==0).values)/w; frac=np.arange(1,n+1)/n
    cg=np.interp(grid,np.concatenate([[0],frac]),np.concatenate([[0],cap]))
    return float(np.trapezoid(cap,frac)),cg
grid=np.linspace(0,1,101)
fig,ax=plt.subplots(figsize=(5.4,3.8)); sg(ax,"both")
ax.plot([0,1],[0,1],color=REF,lw=0.8,ls=(0,(4,3)),zorder=1,label="Random escalation")
B=2000
for m in ORDER:
    temps=[0.0,0.5,1.0,1.5] if m!='Claude' else [0.0,0.5,1.0]
    tabs={t:s2[(s2.model==m)&(s2.temp==t)].reset_index(drop=True) for t in temps}
    # 平均曲線
    curves=[]
    for t in temps:
        _,cg=triage_curve_auc(tabs[t],grid)
        if cg is not None: curves.append(cg)
    avg=np.mean(np.vstack(curves),axis=0)
    # ブートストラップで曲線帯
    boot_curves=[]
    for _ in range(B):
        cs=[]
        for t in temps:
            sub=tabs[t]; idx=np.random.choice(len(sub),len(sub),replace=True)
            _,cg=triage_curve_auc(sub.iloc[idx].reset_index(drop=True),grid)
            if cg is not None: cs.append(cg)
        if cs: boot_curves.append(np.mean(np.vstack(cs),axis=0))
    bc=np.vstack(boot_curves); lo=np.percentile(bc,2.5,axis=0); hi=np.percentile(bc,97.5,axis=0)
    tr=SOT["triage"][m]
    ax.fill_between(grid,lo,hi,color=FILL[m],alpha=0.4,lw=0,zorder=2)
    lab=f"{LAB[m]}  (AUC {tr['AUC']:.2f}, 95% CI {tr['CI'][0]:.2f}\u2013{tr['CI'][1]:.2f})"
    ax.plot(grid,avg,color=COL[m],lw=1.8,zorder=3,label=lab)
    idx=np.linspace(8,92,4).astype(int)
    ax.plot(grid[idx],avg[idx],color=COL[m],marker=MK[m],ms=4,lw=0,mfc=COL[m],mec="white",mew=0.5,zorder=4)
ax.set_xlim(0,1); ax.set_ylim(0,1.02)
ax.xaxis.set_major_locator(MultipleLocator(0.2)); ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.set_xlabel("Fraction of items escalated to human review\n(lowest consistency first)")
ax.set_ylabel("Fraction of model's errors caught")
ax.legend(loc="lower right",frameon=False,fontsize=6.8,handletextpad=0.5)
save(fig,"Figure3")

print("\nStudy2版3図 生成完了（AUC:",{m:SOT['triage'][m]['AUC'] for m in ORDER},"）")
