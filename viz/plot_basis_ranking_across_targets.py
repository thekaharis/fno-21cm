#!/usr/bin/env python3
"""Rank each operator basis on 2-D x_HI, 3-D x_HI and z_re, and connect the ranks.

Only bases that were actually trained on ALL THREE targets can be compared
end-to-end; everything else is shown as an isolated marker so the gaps are
visible rather than silently dropped.

Metric is the task's own: best `val_l2` for the two x_HI targets, best
`val_rmse_z` (from final_report.json) for z_re, whose target is a redshift map
in different units. Ranks are therefore within-target and never compared across
targets as raw numbers. Each basis is represented by its BEST run on that
target, so this reflects the best result achieved, not an average, and the runs
behind each point differ in epochs, loss and tuning effort.
"""
from __future__ import annotations
import csv, json, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS=[("2d_xhi","2-D x_HI","val_l2"),("3d_xhi","3-D x_HI","val_l2"),("zre","z_re","val_rmse_z")]
BASIS={"fno":"Fourier","localfno":"Fourier","ufno":"Fourier","fno_fno":"Fourier",
       "fno_whno":"Fourier + Walsh","whno_fno":"Walsh + Fourier","whno_whno":"Walsh","whno_swhno":"Walsh",
       "swhno_swhno":"SIREN-Walsh","sfno_swhno":"SIREN","localsirenfno":"SIREN","sirenfno":"SIREN",
       "localwno":"Wavelet","wno_whno":"Wavelet","lwf_lwf":"Learned waveform","fno_lwf":"Learned waveform",
       "cnn_fno":"CNN","cnn_sfno":"CNN","cnn_swhno":"CNN","cnn_whno":"CNN"}
SURFACE,INK,INK2,GRID,MUTED="#fcfcfb","#0b0b0b","#52514e","#e4e3de","#b8b6b0"
HUE={"Fourier":"#2a78d6","Wavelet":"#eb6834","Walsh":"#1baf7a","SIREN":"#9d4edd",
     "SIREN-Walsh":"#7b2cbf","Fourier + Walsh":"#00a6a6","Walsh + Fourier":"#0b7a75",
     "Learned waveform":"#d4a017","CNN":"#c2255c"}

def best_metric(run: Path, metric: str):
    if metric=="val_l2":
        m=run/"metrics.jsonl"
        if m.is_file():
            v=[x["val_l2"] for x in (json.loads(l) for l in m.open() if l.strip()) if "val_l2" in x]
            if v: return min(v)
    fr=run/"final_report.json"
    if fr.is_file():
        d=json.loads(fr.read_text())
        if metric in d: return d[metric]
        if "val_l2" in d: return d["val_l2"]
    return None

def collect():
    out={}
    for task,_,metric in TASKS:
        per={}
        root=Path("checkpoints")/task
        for fam in sorted(os.listdir(root)):
            fp=root/fam
            if not fp.is_dir(): continue
            vals=[(v,r) for r in sorted(os.listdir(fp)) if (v:=best_metric(fp/r,metric)) is not None]
            if not vals: continue
            basis=BASIS.get(fam,fam); v,r=min(vals)
            if basis not in per or v<per[basis][0]: per[basis]=(v,f"{fam}/{r}")
        out[task]=per
    return out

def main():
    data=collect()
    common=set.intersection(*(set(data[t]) for t,_,_ in TASKS))
    ranks={t:{b:i+1 for i,(b,_) in enumerate(sorted(data[t].items(), key=lambda kv: kv[1][0]))} for t,_,_ in TASKS}
    allb=sorted(set().union(*(set(data[t]) for t,_,_ in TASKS)))

    fig,ax=plt.subplots(figsize=(10.5,6.4)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    xs={t:i for i,(t,_,_) in enumerate(TASKS)}
    for b in allb:
        pts=[(xs[t],ranks[t][b]) for t,_,_ in TASKS if b in ranks[t]]
        colour=HUE.get(b,MUTED); full=b in common
        if full:
            ax.plot([p[0] for p in pts],[p[1] for p in pts],"-o",color=colour,lw=2.6,ms=11,
                    mec=SURFACE,mew=2,zorder=3,label=b)
        else:
            ax.plot([p[0] for p in pts],[p[1] for p in pts],":o",color=colour,lw=1.2,ms=8,alpha=.55,
                    mec=SURFACE,mew=1.5,zorder=2,label=f"{b} (not on all targets)")
        for x,y in pts:
            v=data[[t for t,_,_ in TASKS][x]][b][0]
            ax.annotate(f"{v:.4f}",(x,y),textcoords="offset points",xytext=(0,-17),ha="center",
                        fontsize=7.2,color=INK2 if full else MUTED,zorder=4)
    ax.set_xticks(list(xs.values())); ax.set_xticklabels([lbl for _,lbl,_ in TASKS],fontsize=11,color=INK)
    ax.set_xlim(-0.45,len(TASKS)-0.3); ax.invert_yaxis()
    ax.set_yticks(range(1,max(len(r) for r in ranks.values())+1))
    ax.set_ylabel("rank within target  (1 = best)",fontsize=10,color=INK2)
    ax.set_title("Operator basis ranking across the three prediction targets",fontsize=13,color=INK,loc="left",pad=26)
    ax.text(0,1.03,"Solid = trained on all three targets (comparable end-to-end).  Dotted = missing from at "
            "least one target.  Number under each point is that basis's best run.",
            transform=ax.transAxes,fontsize=8.5,color=INK2,va="bottom")
    ax.grid(True,axis="y",color=GRID,lw=.8,zorder=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2,labelsize=9)
    ax.legend(loc="center left",bbox_to_anchor=(1.01,.5),frameon=False,fontsize=8.5)
    fig.text(.01,.005,"Metric is each target's own: best val_l2 (x_HI targets), best val_rmse_z (z_re). "
             "Ranks are within-target only. Each basis uses its best run; runs differ in epochs, loss and tuning.",
             fontsize=7.5,color=INK2)
    out=Path("figures/summary/basis_ranking_across_targets.png"); out.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(rect=(0,.02,1,.99)); fig.savefig(out,dpi=160,facecolor=SURFACE,bbox_inches="tight")
    with open(out.with_suffix(".csv"),"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["basis","on_all_targets"]+[f"{t}_{c}" for t,_,_ in TASKS for c in ("rank","best","run")])
        for b in allb:
            row=[b,b in common]
            for t,_,_ in TASKS:
                row += [ranks[t].get(b,""), f"{data[t][b][0]:.6f}" if b in data[t] else "", data[t][b][1] if b in data[t] else ""]
            w.writerow(row)
    print("bases on ALL THREE targets:", sorted(common))
    for t,lbl,metric in TASKS:
        print(f"\n{lbl} (by {metric}):")
        for b,(v,r) in sorted(data[t].items(), key=lambda kv: kv[1][0]):
            print(f"  {ranks[t][b]:>2}. {b:18s} {v:.4f}   {r}{'' if b in common else '   [not on all targets]'}")
    print(f"\nwrote {out} and {out.with_suffix('.csv')}")

if __name__=="__main__": main()
