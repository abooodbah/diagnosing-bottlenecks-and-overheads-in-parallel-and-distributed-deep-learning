# -*- coding: utf-8 -*-
"""Analysis and figures for the sharding and expert-routing measurements.

Reads raw/{pvc,h100}/*.json (llama runs + moe runs), aggregates reps
(median of per-rep step p50s), decomposes step time into exposed
categories from the DeepSpeed comm-op logger, and renders:

  figures/zero_migration.png    step composition, scaling, transport contrast
  figures/moe_imbalance.png     balanced vs skewed all-to-all composition
  processed/summary.csv         per-cell medians and decomposition
"""
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "raw")

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


def load(platform):
    runs = []
    for p in glob.glob(os.path.join(RES, platform, "*.json")):
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
    """v2 runs carry synchronized comm timers. Explicit tag when present;
    otherwise the v2 reruns used non-default iteration counts, and their
    per-rank comm totals are a macroscopic share of the step."""
    if run.get("bench_version") == 2:
        return True
    if run.get("iters") != 40:
        return True
    ranks = [c for c in (run.get("comms_all_ranks") or []) if c]
    if not ranks:
        return False
    per_rank = (sum(sum(v["time_ms"] for v in c.values()) for c in ranks)
                / len(ranks) / run["iters"])
    return per_rank / run["step_max_ms"]["p50"] > 0.10


def comm_ms_per_iter(run, ops):
    """Mean over ranks of summed op wall time, per measured iteration."""
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
    # prefer v2 runs (synchronized comm timers) for the decomposition,
    # and among those prefer runs that also carry deepspeed.comm-level
    # buckets (full ZeRO-3 param-fetch attribution)
    v2 = [r for r in sel if _is_v2(r)]
    pool = v2 or sel
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
    return d


def main():
    out_rows = []
    data = {}
    for plat in ("pvc", "h100"):
        rr = llama_runs(load(plat))
        data[plat] = rr
        for n in (1, 2, 4):
            for m in MODES:
                a = agg(rr, n, m)
                if a:
                    out_rows.append((plat, n, m, a))

    with open(os.path.join(os.path.dirname(HERE), "processed", "summary.csv"), "w") as f:
        f.write("platform,nodes,mode,reps,step_p50_ms,spread_ms,peak_gib,"
                "compute_ms,grad_reduce_ms,param_gather_ms,sync_ms,other_comm_ms\n")
        for plat, n, m, a in out_rows:
            f.write(f"{plat},{n},{m},{a['reps']},{a['step_med']:.1f},"
                    f"{a['step_spread']:.1f},{a['peak_gib']:.1f},"
                    f"{a['compute']:.1f},{a['grad']:.1f},{a['gather']:.1f},"
                    f"{a['sync']:.1f},{a['othercomm']:.1f}\n")
    print("wrote zmig_summary.csv")
    for row in out_rows:
        plat, n, m, a = row
        print(f"{plat} n{n} {m:6s} reps={a['reps']} step={a['step_med']:8.1f}ms "
              f"mem={a['peak_gib']:5.1f}GiB comp={a['compute']:7.1f} "
              f"grad={a['grad']:7.1f} gath={a['gather']:7.1f} sync={a['sync']:6.1f}")

    # ---------------- figure: panel (a) + (b) + (c) --------------------
    # canvas widened for panel (c); FS scaled so text renders at the same
    # size once LaTeX scales the figure to text width
    FS = 14.5
    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(9.7, 3.3),
        gridspec_kw={"width_ratios": [1.60, 1.05, 0.62]})

    target_n = 4
    bars, labels, plats = [], [], []
    for plat in ("pvc", "h100"):
        for m in MODES:
            a = agg(data[plat], target_n, m)
            bars.append(a)
            labels.append(MODE_LABEL[m])
            plats.append(plat)
    have = [i for i, b in enumerate(bars) if b]
    xs = range(len(have))
    cats = [("compute", C_COMPUTE, "Compute + unattributed"),
            ("grad", C_GRAD, "Gradient reduction"),
            ("gather", C_GATHER, "Parameter all-gather"),
            ("sync", C_SYNC, "Synchronization"),
            ("othercomm", C_OTHER, "Other comm")]
    # normalize within platform to its DDP step
    base = {}
    for plat in ("pvc", "h100"):
        d = agg(data[plat], target_n, "ddp")
        base[plat] = d["step"] if d else None
    tops = []
    for i in have:
        b = bars[i]
        bn = base[plats[i]] or b["step"]
        tops.append(sum(b[k] for k, _, _ in cats) / bn)
    # If one bar dwarfs the rest (H100 ZeRO-3), cap the axis and let that
    # bar run off the top with break marks and its true multiple annotated,
    # so the other bars' compositions stay readable.
    ycap = 4.8 if tops and max(tops) > 6 else None
    for xi, i in enumerate(xs):
        b = bars[have[i]]
        plat = plats[have[i]]
        bn = base[plat] or b["step"]
        bottom = 0.0
        for key, color, lab in cats:
            v = b[key] / bn
            axa.bar(xi, v, 0.62, bottom=bottom, color=color,
                    edgecolor="#333333", linewidth=0.6,
                    label=lab if xi == 0 else None)
            bottom += v
        if ycap and bottom > ycap:
            for yb in (ycap * 0.90, ycap * 0.935):
                axa.plot([xi - 0.36, xi + 0.36], [yb - 0.09, yb + 0.09],
                         color="white", linewidth=2.8, zorder=5,
                         solid_capstyle="butt")
                axa.plot([xi - 0.36, xi + 0.36], [yb - 0.09, yb + 0.09],
                         color="#333333", linewidth=0.9, zorder=6)
            axa.text(xi, ycap * 0.55, f"{bottom:.0f}x", ha="center",
                     fontsize=FS - 1.8, fontweight="bold", color="#333333",
                     bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                               edgecolor="none", alpha=0.85), zorder=7)
            axa.text(xi, ycap + 0.07, f"{b['peak_gib']:.0f}", ha="center",
                     fontsize=FS - 2.2)
        else:
            axa.text(xi, bottom + 0.03, f"{b['peak_gib']:.0f}", ha="center",
                     fontsize=FS - 2.2)
    if tops:
        axa.set_ylim(0, ycap if ycap else max(tops) * 1.18)
    axa.set_xticks(list(xs))
    axa.set_xticklabels([labels[i] for i in have], fontsize=FS - 4.2)
    n_pvc = sum(1 for i in have if plats[i] == "pvc")
    if n_pvc and n_pvc < len(have):
        axa.axvline(n_pvc - 0.5, color="#888888", linewidth=0.8, ls=":")
    if n_pvc:
        axa.text((n_pvc - 1) / 2, -0.16, PLAT_LABEL["pvc"], ha="center",
                 transform=axa.get_xaxis_transform(), fontsize=FS - 2.6)
    if n_pvc < len(have):
        axa.text((n_pvc + len(have) - 1) / 2, -0.16, PLAT_LABEL["h100"],
                 ha="center", transform=axa.get_xaxis_transform(),
                 fontsize=FS - 2.6)
    axa.set_ylabel("Step time vs platform DDP", fontsize=FS - 2)
    axa.set_xlabel("(a)  four-node composition; number = peak GiB/rank",
                   fontsize=FS - 1, labelpad=18)
    axa.legend(fontsize=FS - 4.0, frameon=True, edgecolor="#bbbbbb",
               loc="upper left", handlelength=1.3, labelspacing=0.26,
               borderpad=0.3, handletextpad=0.45)

    # panel (b): scaling
    for plat, marker in (("pvc", "o"), ("h100", "s")):
        rr = data[plat]
        d1 = agg(rr, 1, "ddp")
        if not d1:
            continue
        for m, ls, col in (("ddp", "-", "#0072B2"),
                           ("zero2", "-.", "#009E73"),
                           ("zero3", "--", "#D55E00")):
            ns, ys, es = [], [], []
            for n in (1, 2, 4):
                a = agg(rr, n, m)
                if a:
                    thr1 = d1["world"] * 2048 / (d1["step_med"] / 1e3)
                    thr = a["world"] * 2048 / (a["step_med"] / 1e3)
                    ns.append(n)
                    ys.append(thr / thr1)
                    # first-order spread propagation: dy/y = dstep/step
                    es.append(thr / thr1 * a["step_spread"] / a["step_med"]
                              if a["step_spread"] else 0.0)
            if ns:
                axb.errorbar(ns, ys, yerr=es, fmt=ls, marker=marker, color=col,
                             lw=1.3, ms=4.5, mfc="white", capsize=2,
                             elinewidth=0.8,
                             label=f"{PLAT_LABEL[plat].split()[0]} {MODE_LABEL[m]}")
    axb.plot([1, 4], [1, 4], ":", color="#999999", lw=0.9, label="Ideal")
    axb.set_xticks([1, 2, 4])
    axb.set_xlabel("(b)  nodes", fontsize=FS - 1)
    axb.set_ylabel("Throughput vs own 1-node DDP", fontsize=FS - 2.4)
    axb.legend(fontsize=FS - 4.4, frameon=True, edgecolor="#bbbbbb",
               loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.3,
               labelspacing=0.25, borderpad=0.3, handletextpad=0.4)
    # panel (c): ZeRO-3 transport contrast at 2 nodes
    cbars = []
    p2 = agg(data["pvc"], 2, "zero3")
    if p2:
        cbars.append(("Intel\nTCP", p2))
    h2 = agg(data["h100"], 2, "zero3")
    if h2:
        cbars.append(("H100\nIB", h2))
    h2s = agg(data["h100"], 2, "zero3", transport="tcp-sockets")
    if h2s:
        cbars.append(("H100\nTCP", h2s))
    for xi, (lab, b) in enumerate(cbars):
        axc.bar(xi, b["step_med"] / 1e3, 0.58, color=C_GATHER,
                edgecolor="#333333", linewidth=0.7,
                yerr=b["step_spread"] / 1e3 if b["step_spread"] else None,
                capsize=3)
        axc.text(xi, (b["step_med"] + b["step_spread"]) / 1e3 + 0.7,
                 f"{b['peak_gib']:.0f}", ha="center", fontsize=FS - 2.6)
    axc.set_xticks(range(len(cbars)))
    axc.set_xticklabels([c[0] for c in cbars], fontsize=FS - 3.4)
    axc.set_xlim(-0.6, len(cbars) - 0.4)
    axc.set_ylim(0, 24)
    axc.set_ylabel("ZeRO-3 step (s)", fontsize=FS - 2)
    axc.set_xlabel("(c)  2-node transport", fontsize=FS - 1)
    for ax in (axa, axb, axc):
        ax.tick_params(labelsize=FS - 1.8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.4, color="#dddddd")
    fig.subplots_adjust(left=0.060, right=0.988, top=0.97, bottom=0.26,
                        wspace=0.40)
    fig.savefig(os.path.join(os.path.dirname(HERE), "figures", "zero_migration.png"), dpi=300)
    print("wrote figures/zero_migration.png")

    # ------------- transport contrast (ZeRO-3, 2 nodes) ---------------
    tbars = []
    p = agg(data["pvc"], 2, "zero3")
    if p:
        tbars.append(("Intel\ntcp (IPoIB)", p))
    hib = agg(data["h100"], 2, "zero3") or agg(data["h100"], 2, "zero3",
                                               transport="host-staged")
    if hib:
        tbars.append(("H100\nIB, host-staged", hib))
    hso = agg(data["h100"], 2, "zero3", transport="tcp-sockets")
    if hso:
        tbars.append(("H100\nTCP sockets", hso))
    if len(tbars) >= 2:
        fig3, ax3 = plt.subplots(figsize=(4.6, 2.9))
        for xi, (lab, b) in enumerate(tbars):
            ax3.bar(xi, b["step_med"] / 1e3, 0.55, color="#5b8db8",
                    edgecolor="#333333", linewidth=0.7,
                    yerr=b["step_spread"] / 1e3 if b["step_spread"] else None,
                    capsize=3)
            ax3.text(xi, (b["step_med"] + b["step_spread"]) / 1e3 + 0.6,
                     f"{b['peak_gib']:.0f} GiB", ha="center", fontsize=FS - 2.4)
        ax3.set_xticks(range(len(tbars)))
        ax3.set_xticklabels([t[0] for t in tbars], fontsize=FS - 2)
        ax3.set_ylabel("ZeRO-3 step p50 (s), 2 nodes", fontsize=FS - 1.5)
        ax3.set_ylim(0, max(b["step_med"] for _, b in tbars) / 1e3 * 1.22)
        ax3.tick_params(labelsize=FS - 2)
        ax3.set_axisbelow(True)
        ax3.grid(axis="y", linewidth=0.4, color="#dddddd")
        fig3.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.18)
        fig3.savefig(os.path.join(os.path.dirname(HERE), "figures", "transport_contrast.png"), dpi=200)
        print("wrote figures/transport_contrast.png")

    # ---------------- MoE figure --------------------------------------
    plats_with_data = [p for p in ("pvc", "h100") if moe_runs(load(p))]
    if plats_with_data:
        n = len(plats_with_data)
        fig2, axes = plt.subplots(1, n, figsize=(3.7 * n, 2.9), squeeze=False)
        axes = axes[0]
        for ax, plat in zip(axes, plats_with_data):
            tops = []
            mm = moe_runs(load(plat))
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
        fig2.savefig(os.path.join(os.path.dirname(HERE), "figures", "moe_imbalance.png"), dpi=300)
        print("wrote figures/moe_imbalance.png")


if __name__ == "__main__":
    main()
