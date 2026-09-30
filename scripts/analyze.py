# -*- coding: utf-8 -*-
"""Analysis and figures for the sharding and expert-routing measurements.

Reads raw/{pvc,h100}/*.json (llama runs + moe runs), aggregates reps
(median of per-rep step p50s), decomposes step time into exposed categories
from the harness's synchronized timers, and renders:

  figures/zero_migration.png    (a) four-node step composition, (b) DDP and
                                ZeRO-2 weak scaling, (c) ZeRO-3 step time at
                                each logged iteration against the training step
  figures/transport_contrast.png  two-node H100 ZeRO-3, InfiniBand against TCP,
                                at each logged iteration
  figures/moe_imbalance.png     balanced vs skewed all-to-all composition
  processed/summary.csv         per-cell medians and decomposition
  processed/figure_data.json    every value the figures plot

ZeRO-3 step time grows by about 0.55 s (H100) and 0.28 s (Intel) per step
during every run because of DeepSpeed's communication logger, and the Intel
ZeRO-3 cells were measured over windows of different length, so ZeRO-3 is
shown per iteration rather than as window medians or scaling ratios. The
per-iteration values come from processed/drift_launches.csv (scripts/drift.py).

Records whose communication fields hold DeepSpeed logger records (the 6 Sep
harness) have no valid times; they are never used for the decomposition.
Options --raw and --out point the script at another copy of the records and
another output folder (used by scripts/check_time_ms.py).
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import comm_layout  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["svg.hashsalt"] = "where-the-cost-lands"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "raw")

GRAD_OPS = ("all_reduce", "reduce_scatter_tensor", "reduce_scatter",
            "reduce_scatter_fn", "reduce", "inference_all_reduce")
GATHER_OPS = ("all_gather_into_tensor", "all_gather", "allgather_base",
              "allgather_fn", "all_gather_base")
SYNC_OPS = ("barrier",)

MODES = ["ddp", "zero2", "zero3"]
MODE_LABEL = {"ddp": "DDP", "zero2": "ZeRO-2", "zero3": "ZeRO-3"}
PLAT_LABEL = {"pvc": "Intel Max 1550", "h100": "NVIDIA H100"}

C_COMPUTE = "#5b8db8"
C_GRAD = "#7fcdbb"
C_GATHER = "#f5d76e"
C_SYNC = "#f2b27e"
C_OTHER = "#c8c8c8"
# ZeRO-3 trajectories: one colour per platform and node count
TRAJ = {("h100", 1): "#fdae61", ("h100", 2): "#f46d43", ("h100", 4): "#a50026",
        ("pvc", 1): "#91bfdb", ("pvc", 2): "#4575b4", ("pvc", 4): "#313695"}


def load(platform, raw=None):
    runs = []
    for p in sorted(glob.glob(os.path.join(raw or RES, platform, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        d["_file"] = os.path.basename(p)
        runs.append(d)
    return runs


def llama_runs(runs):
    return [r for r in runs if "mode" in r and r.get("mode") in MODES]


def moe_runs(runs):
    return [r for r in runs if r.get("mode") in ("balanced", "skewed")]


def _is_v2(run):
    """Runs whose communication fields are the harness's synchronized timers.
    Decided by the record layout (common.comm_layout), never by the size of
    the time values: in the 6 Sep records 'time_ms' is not a time."""
    return comm_layout(run) == "harness-timers"


def comm_ms_per_iter(run, ops):
    """Mean over ranks of summed op wall time, per measured iteration."""
    if not _is_v2(run):
        raise ValueError("%s: communication fields hold no valid times" % run.get("_file"))
    total = []
    for c in run.get("comms_all_ranks") or []:
        if not c:
            continue
        total.append(sum(v["time_ms"] for k, v in c.items() if k in ops))
    if not total:
        return 0.0
    return (sum(total) / len(total)) / run["iters"]


def decompose(run):
    step = run["step_max_ms"]["p50"]
    grad = comm_ms_per_iter(run, GRAD_OPS)
    gather = comm_ms_per_iter(run, GATHER_OPS)
    syncb = st.median(run["barrier_ms_mean_per_rank"])
    allops = comm_ms_per_iter(
        run, tuple({k for c in (run.get("comms_all_ranks") or [{}]) if c for k in c}))
    other_comm = max(0.0, allops - grad - gather
                     - comm_ms_per_iter(run, SYNC_OPS))
    parts = {"grad": grad, "gather": gather, "sync": syncb, "othercomm": other_comm}
    ssum = sum(parts.values())
    if ssum > step:  # overlapped ops exceed wall step: scale to fit
        parts = {k: v * step / ssum for k, v in parts.items()}
        parts["compute"] = 0.0
    else:
        parts["compute"] = step - ssum
    parts["step"] = step
    return parts


def agg(platform_runs, nodes, mode, transport=None):
    def tnote(r):
        return (r.get("env", {}).get("transport_note", "") or "") + "|" + r.get("tag", "")

    if transport is None:
        # main matrix: the platform's qualified default path only
        sel = [r for r in platform_runs
               if r["mode"] == mode and r["nodes"] == nodes
               and "host-staged" not in r.get("tag", "")
               and "tcp-sockets" not in r.get("tag", "")]
    else:
        sel = [r for r in platform_runs
               if r["mode"] == mode and r["nodes"] == nodes
               and transport in tnote(r)]
    if not sel:
        return None
    steps = [r["step_max_ms"]["p50"] for r in sel]
    # the decomposition uses runs with valid synchronized comm timers, and
    # among those prefers runs that also carry deepspeed.comm-level buckets
    v2 = [r for r in sel if _is_v2(r)]
    if not v2:
        raise ValueError("no run with valid comm times for %s n%d %s" % (mode, nodes, transport))
    pool = v2
    ds = [r for r in pool
          if any("allgather_fn" in (c or {})
                 for c in (r.get("comms_all_ranks") or []))]
    pool = ds or pool
    med = sorted(pool, key=lambda r: r["step_max_ms"]["p50"])[len(pool) // 2]
    d = decompose(med)
    d["step_med"] = st.median(steps)
    d["step_spread"] = (max(steps) - min(steps)) / 2 if len(steps) > 1 else 0.0
    d["reps"] = len(sel)
    d["peak_gib"] = max(r["peak_mem_bytes_max"] for r in sel) / 2**30
    d["world"] = med["world"]
    d["decomposed_file"] = med["_file"]
    return d


def zero3_trajectories(processed):
    """Kept ZeRO-3 launches with their logged step times, from drift.py."""
    path = os.path.join(processed, "drift_launches.csv")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "processed", "drift_launches.csv")
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["mode"] != "zero3" or not r["result_status"].startswith("kept in raw/"):
            continue
        out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RES)
    ap.add_argument("--out", default=ROOT, help="folder that receives processed/ and figures/")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    processed = os.path.join(args.out, "processed")
    figures = os.path.join(args.out, "figures")
    os.makedirs(processed, exist_ok=True)
    os.makedirs(figures, exist_ok=True)

    out_rows = []
    data = {}
    for plat in ("pvc", "h100"):
        rr = llama_runs(load(plat, args.raw))
        data[plat] = rr
        for n in (1, 2, 4):
            for m in MODES:
                a = agg(rr, n, m)
                if a:
                    out_rows.append((plat, n, m, a))

    with open(os.path.join(processed, "summary.csv"), "w", newline="\n") as f:
        f.write("platform,nodes,mode,reps,step_p50_ms,spread_ms,peak_gib,"
                "compute_ms,grad_reduce_ms,param_gather_ms,sync_ms,other_comm_ms\n")
        for plat, n, m, a in out_rows:
            f.write(f"{plat},{n},{m},{a['reps']},{a['step_med']:.1f},"
                    f"{a['step_spread']:.1f},{a['peak_gib']:.1f},"
                    f"{a['compute']:.1f},{a['grad']:.1f},{a['gather']:.1f},"
                    f"{a['sync']:.1f},{a['othercomm']:.1f}\n")
    print("wrote processed/summary.csv")
    fig_data = {"summary": [{"platform": plat, "nodes": n, "mode": m,
                             **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()}}
                            for plat, n, m, a in out_rows]}
    if args.no_figures:
        with open(os.path.join(processed, "figure_data.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(fig_data, f, indent=1, sort_keys=True)
        return 0

    FS = 14.5
    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(9.7, 3.3),
        gridspec_kw={"width_ratios": [1.45, 0.95, 1.25]})

    # (a) four-node composition, as shares of each configuration's own step
    cats = [("compute", C_COMPUTE, "Compute + unattributed"),
            ("grad", C_GRAD, "Gradient reduction"),
            ("gather", C_GATHER, "Parameter all-gather"),
            ("sync", C_SYNC, "Synchronization"),
            ("othercomm", C_OTHER, "Other comm")]
    cells = [(p, m) for p in ("pvc", "h100") for m in MODES]
    comp = []
    for xi, (plat, m) in enumerate(cells):
        b = agg(data[plat], 4, m)
        tot = sum(b[k] for k, _, _ in cats)
        bottom = 0.0
        shares = {}
        for key, color, lab in cats:
            v = 100.0 * b[key] / tot
            shares[key] = round(v, 3)
            axa.bar(xi, v, 0.62, bottom=bottom, color=color, edgecolor="#333333",
                    linewidth=0.6, label=lab if xi == 0 else None)
            bottom += v
        axa.text(xi, 102, f"{b['peak_gib']:.0f}", ha="center", fontsize=FS - 2.2)
        comp.append({"platform": plat, "mode": m, "shares_percent": shares,
                     "peak_gib": round(b["peak_gib"], 2), "decomposed_file": b["decomposed_file"]})
    fig_data["composition_4node"] = comp
    axa.set_ylim(0, 112)
    axa.set_xticks(range(len(cells)))
    axa.set_xticklabels([{"ddp": "DDP", "zero2": "Z2", "zero3": "Z3"}[m] for _, m in cells], fontsize=FS - 2.8)
    axa.axvline(2.5, color="#888888", linewidth=0.8, ls=":")
    axa.text(1, -0.19, PLAT_LABEL["pvc"], ha="center", transform=axa.get_xaxis_transform(), fontsize=FS - 3.2)
    axa.text(4, -0.19, PLAT_LABEL["h100"], ha="center", transform=axa.get_xaxis_transform(), fontsize=FS - 3.2)
    axa.set_ylabel("Share of step (%)", fontsize=FS - 2)
    axa.set_xlabel("(a)  four-node composition; number = peak GiB/rank", fontsize=FS - 1, labelpad=18)
    axa.text(0.5, -0.335, "gather bars are timed intervals and can include logger work",
             transform=axa.transAxes, ha="center", va="top", fontsize=FS - 5.4, color="#555555")
    axa.legend(fontsize=FS - 4.6, frameon=True, edgecolor="#bbbbbb", loc="lower left",
               bbox_to_anchor=(0.0, 1.02), ncol=3, handlelength=1.1, labelspacing=0.2,
               borderpad=0.25, handletextpad=0.35, columnspacing=0.6)

    # (b) weak scaling of DDP and ZeRO-2 (their step time is stable)
    scal = []
    for plat, marker in (("pvc", "o"), ("h100", "s")):
        rr = data[plat]
        d1 = agg(rr, 1, "ddp")
        for m, ls, col in (("ddp", "-", "#0072B2"), ("zero2", "-.", "#009E73")):
            ns, ys, es = [], [], []
            for n in (1, 2, 4):
                a = agg(rr, n, m)
                thr1 = d1["world"] * 2048 / (d1["step_med"] / 1e3)
                thr = a["world"] * 2048 / (a["step_med"] / 1e3)
                ns.append(n)
                ys.append(thr / thr1)
                es.append(thr / thr1 * a["step_spread"] / a["step_med"] if a["step_spread"] else 0.0)
            axb.errorbar(ns, ys, yerr=es, fmt=ls, marker=marker, color=col, lw=1.3, ms=4.5,
                         mfc="white", capsize=2, elinewidth=0.8,
                         label=f"{'Intel' if plat == 'pvc' else 'H100'} {MODE_LABEL[m]}")
            scal.append({"platform": plat, "mode": m, "nodes": ns,
                         "throughput_vs_1node_ddp": [round(y, 4) for y in ys]})
    fig_data["weak_scaling"] = scal
    axb.plot([1, 4], [1, 4], ":", color="#999999", lw=0.9, label="Ideal")
    axb.set_xticks([1, 2, 4])
    axb.set_xlabel("(b)  nodes", fontsize=FS - 1)
    axb.set_ylabel("Throughput vs own 1-node DDP", fontsize=FS - 2.4)
    axb.legend(fontsize=FS - 4.8, frameon=True, edgecolor="#bbbbbb", loc="upper left", ncol=1,
               handlelength=1.3, labelspacing=0.2, borderpad=0.3, handletextpad=0.4)

    # (c) ZeRO-3 step time at each logged iteration, against the training step
    traj = zero3_trajectories(processed)
    seen = set()
    tj = []
    for r in sorted(traj, key=lambda r: (r["platform"], int(r["nodes"]), r["transport"], r["rep"])):
        w = int(r["warmup_iters"])
        pts = [(w + int(a_), int(b_) / 1e3) for a_, b_ in (x.split(":") for x in r["heartbeats_ms"].split(";"))]
        plat, n = r["platform"], int(r["nodes"])
        tcp = r["transport"] == "sock"
        host = r["transport"] == "host"
        key = (plat, n, "tcp" if tcp else ("host" if host else ""))
        label = None
        if key not in seen:
            seen.add(key)
            sfx = ", TCP" if tcp else (", IB (earlier job)" if host else (", IB" if (plat == "h100" and n > 1) else ""))
            label = "%s, %d node%s%s" % ("H100" if plat == "h100" else "Intel", n, "" if n == 1 else "s", sfx)
        axc.plot([p[0] for p in pts], [p[1] for p in pts], ls="--" if tcp else (":" if host else "-"),
                 marker="s" if plat == "h100" else "o", ms=3.2, lw=1.0, color=TRAJ[(plat, n)],
                 mfc="white" if tcp else TRAJ[(plat, n)], label=label)
        tj.append({"platform": plat, "nodes": n, "transport": r["transport"], "rep": r["rep"],
                   "job": r["job"], "warmup": w, "measured": int(r["measured_iters"]),
                   "points_step_s": [[p[0], p[1]] for p in pts]})
    fig_data["zero3_trajectories"] = tj
    axc.axvline(11, color="#888888", lw=0.7, ls=":")
    axc.text(11.6, 1.2, "iteration 11", fontsize=FS - 5, color="#555555")
    axc.set_xlim(0, 44)
    axc.set_ylim(0, 29)
    axc.set_xlabel("(c)  ZeRO-3, iteration index (warm-up + measured)", fontsize=FS - 1)
    axc.set_ylabel("Step time (s)", fontsize=FS - 2)
    axc.legend(fontsize=FS - 6.2, frameon=True, edgecolor="#bbbbbb", loc="lower left",
               bbox_to_anchor=(-0.02, 1.02), ncol=2, handlelength=1.5, labelspacing=0.12, borderpad=0.25,
               handletextpad=0.3, columnspacing=0.6)
    for ax in (axa, axb, axc):
        ax.tick_params(labelsize=FS - 1.8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.4, color="#dddddd")
    fig.subplots_adjust(left=0.060, right=0.988, top=0.80, bottom=0.26, wspace=0.34)
    fig.savefig(os.path.join(figures, "zero_migration.png"), dpi=300)
    print("wrote figures/zero_migration.png")

    # ------------- transport contrast (H100 ZeRO-3, 2 nodes) ---------------
    fig3, ax3 = plt.subplots(figsize=(4.6, 2.9))
    gaps = []
    ib = {r["rep"]: r for r in traj if r["platform"] == "h100" and r["nodes"] == "2" and r["transport"] == "direct"}
    tcp = {r["rep"]: r for r in traj if r["platform"] == "h100" and r["nodes"] == "2" and r["transport"] == "sock"}
    for rep in sorted(ib):
        for r, col, ls, lab in ((ib[rep], "#f46d43", "-", "InfiniBand, host-staged"),
                                (tcp.get(rep), "#6a51a3", "--", "TCP sockets")):
            if r is None:
                continue
            w = int(r["warmup_iters"])
            pts = [(w + int(a), int(b) / 1e3) for a, b in (x.split(":") for x in r["heartbeats_ms"].split(";"))]
            ax3.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker="o", ms=3, lw=1.0, color=col,
                     label=lab if rep == min(ib) else None)
        if rep in tcp:
            a_ = dict((int(a), int(b)) for a, b in (x.split(":") for x in ib[rep]["heartbeats_ms"].split(";")))
            b_ = dict((int(a), int(b)) for a, b in (x.split(":") for x in tcp[rep]["heartbeats_ms"].split(";")))
            gaps += [b_[k] - a_[k] for k in a_ if k in b_]
    fig_data["transport_gap_ms"] = {"min": min(gaps), "max": max(gaps), "n": len(gaps)}
    ax3.set_xlabel("training step", fontsize=FS - 3)
    ax3.set_ylabel("H100 ZeRO-3 step (s), 2 nodes", fontsize=FS - 3.5)
    ax3.set_xlim(0, 44)
    ax3.set_ylim(0, 29)
    ax3.text(26, 5, "TCP adds %.1f to %.1f s\nat every logged step" % (min(gaps) / 1e3, max(gaps) / 1e3),
             fontsize=FS - 4.5, ha="center")
    ax3.legend(fontsize=FS - 5, frameon=False, loc="upper left")
    ax3.tick_params(labelsize=FS - 3)
    ax3.set_axisbelow(True)
    ax3.grid(axis="y", linewidth=0.4, color="#dddddd")
    fig3.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.18)
    fig3.savefig(os.path.join(figures, "transport_contrast.png"), dpi=200)
    print("wrote figures/transport_contrast.png")

    # ---------------- MoE figure --------------------------------------
    plats_with_data = [p for p in ("pvc", "h100") if moe_runs(load(p, args.raw))]
    moe_data = []
    if plats_with_data:
        n = len(plats_with_data)
        fig2, axes = plt.subplots(1, n, figsize=(3.7 * n, 2.9), squeeze=False)
        axes = axes[0]
        for ax, plat in zip(axes, plats_with_data):
            tops = []
            mm = moe_runs(load(plat, args.raw))
            for xi, mode in enumerate(("balanced", "skewed")):
                sel = [r for r in mm if r["mode"] == mode]
                if not sel:
                    continue
                med = sorted(sel, key=lambda r: r["step_max_ms"]["p50"])[len(sel) // 2]
                pr = med["per_rank_p50"]
                # critical-rank decomposition: stack the per-phase medians of
                # the single rank with the largest summed phase medians, so
                # every segment describes the same rank's iteration
                world = med["world"]
                sums = [pr["dispatch_ms"][i] + pr["expert_ms"][i]
                        + pr["return_ms"][i] + pr["end_wait_ms"][i]
                        for i in range(world)]
                ci = max(range(world), key=lambda i: sums[i])
                bottom = 0.0
                for v, c, lab in (
                        (pr["dispatch_ms"][ci], C_GATHER, "Dispatch a2a"),
                        (pr["expert_ms"][ci], C_COMPUTE, "Expert"),
                        (pr["return_ms"][ci], "#e8b84b", "Return a2a"),
                        (pr["end_wait_ms"][ci], C_SYNC, "End wait")):
                    ax.bar(xi, v, 0.5, bottom=bottom, color=c,
                           edgecolor="#333333", linewidth=0.6,
                           label=lab if (xi == 0 and plat == plats_with_data[0])
                           else None)
                    bottom += v
                # measured step totals drawn as separate markers, not implied
                # by the stack: p50 tick and p50-p95 whisker
                p50 = med["step_max_ms"]["p50"]
                p95 = med["step_max_ms"]["p95"]
                ax.plot([xi - 0.31, xi + 0.31], [p50, p50], color="#222222",
                        lw=1.1, zorder=6,
                        label="Step p50" if (xi == 0 and
                                             plat == plats_with_data[0])
                        else None)
                ax.plot([xi, xi], [p50, p95], color="#222222", lw=0.9,
                        zorder=6)
                ax.plot([xi - 0.10, xi + 0.10], [p95, p95], color="#222222",
                        lw=0.9, zorder=6)
                tops.append(max(bottom, p95))
                ax.text(xi + 0.34, p95, f"p95 {p95:.0f}", ha="left",
                        va="center", fontsize=FS - 2.8)
                moe_data.append({"platform": plat, "mode": mode, "median_rep_file": med["_file"],
                                 "critical_rank": ci, "dispatch_ms": pr["dispatch_ms"][ci],
                                 "expert_ms": pr["expert_ms"][ci], "return_ms": pr["return_ms"][ci],
                                 "end_wait_ms": pr["end_wait_ms"][ci], "p50_ms": p50, "p95_ms": p95})
            ranks = {r["world"] for r in mm}
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Balanced", "Skewed"], fontsize=FS - 1.5)
            ax.set_xlim(-0.55, 1.95)
            ax.set_title(f"{PLAT_LABEL[plat]} (2 nodes, {max(ranks)} ranks)",
                         fontsize=FS - 1.4)
            ax.set_ylabel("ms per iteration", fontsize=FS - 1.5)
            ax.tick_params(labelsize=FS - 2)
            ax.set_axisbelow(True)
            ax.grid(axis="y", linewidth=0.4, color="#dddddd")
            ax.set_ylim(0, max(tops) * 1.18)
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig2.legend(handles, labels_, fontsize=FS - 2.8, loc="upper center",
                    ncol=5, frameon=False, handlelength=1.2,
                    columnspacing=1.0, bbox_to_anchor=(0.5, 1.005))
        fig2.subplots_adjust(left=0.15 if n == 1 else 0.10, right=0.985,
                             top=0.80, bottom=0.13, wspace=0.28)
        fig2.savefig(os.path.join(figures, "moe_imbalance.png"), dpi=300)
        print("wrote figures/moe_imbalance.png")
    fig_data["moe_critical_rank"] = moe_data
    with open(os.path.join(processed, "figure_data.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(fig_data, f, indent=1, sort_keys=True)
    print("wrote processed/figure_data.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
