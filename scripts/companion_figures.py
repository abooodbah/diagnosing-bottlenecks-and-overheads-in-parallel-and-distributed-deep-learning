# -*- coding: utf-8 -*-
"""Figures 1 to 3 of the companion report, from the records in this
repository.

  fig_paths.pdf          measurement setting (schematic)
  fig_zero_expanded.pdf  (a) DDP and ZeRO-2 median step time, (b) peak
                         memory, (c) four-node composition, (d) ZeRO-3 step
                         time at each logged iteration against the training
                         step (processed/drift_launches.csv)
  fig_moe_expanded.pdf   expert routing
  companion_figure_data.json  the values plotted in Figs. 2 and 3

ZeRO-3 is not shown as window medians: its step time grows during every run
and the Intel windows differ in length (processed/launch_windows.csv).
PDFs are written without a creation date so reruns are byte-identical.

usage: python companion_figures.py [--out figures/companion]
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402
from common import PROCESSED  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
PDF_META = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}

PLAT = {"pvc": "Intel Max 1550", "h100": "NVIDIA H100"}
MODE = {"ddp": "DDP", "zero2": "ZeRO-2", "zero3": "ZeRO-3"}
COL = {"ddp": "#0072B2", "zero2": "#009E73", "zero3": "#D55E00"}
MK = {"pvc": "o", "h100": "s"}
LS = {"pvc": "-", "h100": "--"}


def fig_paths(out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.95), gridspec_kw={"width_ratios": [1.2, 1.0]})
    for a in (ax, bx):
        a.axis("off")
        a.set_xlim(0, 100)
        a.set_ylim(-4, 100)

    def box(a, x, y, w, h, title, lines, fc, fs=6.7, step=8.2):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                                   linewidth=0.7, edgecolor="#444444", facecolor=fc))
        a.text(x + w / 2, y + h - 6.5, title, ha="center", va="center", fontsize=fs + 0.8, fontweight="bold")
        for i, t in enumerate(lines):
            a.text(x + w / 2, y + h - 15 - step * i, t, ha="center", va="center", fontsize=fs)

    box(ax, 1, 52, 30, 46, "DDP (stage 0)", ["all state replicated", "observed per step:", "5 all-reduces"], "#e8eef5")
    box(ax, 35, 52, 30, 46, "ZeRO-2", ["gradients, optimizer", "state partitioned;", "observed: 13 all-",
                                       "reduces, 1 all-gather"], "#dcead9")
    box(ax, 69, 52, 30, 46, "ZeRO-3", ["parameters also", "partitioned; observed:", "206 all-gathers,",
                                       "11 reduce-scatters"], "#f6d7cd")
    ax.add_patch(FancyBboxPatch((1, 4), 98, 42, boxstyle="round,pad=0.4,rounding_size=1.4",
                                linewidth=0.7, edgecolor="#444444", facecolor="#f7f7f7"))
    ax.text(50, 40.5, "Timed interval of one collective", ha="center", va="center", fontsize=7.5, fontweight="bold")
    y0, h0 = 21.0, 7.0
    ax.add_patch(plt.Rectangle((6, y0), 32, h0, facecolor="#c8c8c8", edgecolor="#555555", lw=0.6))
    ax.text(16.5, y0 + h0 / 2, "earlier device work", ha="center", va="center", fontsize=6.0)
    ax.add_patch(plt.Rectangle((38, y0), 26, h0, facecolor="#f5d76e", edgecolor="#555555", lw=0.6))
    ax.text(51, y0 + h0 / 2, "collective", ha="center", va="center", fontsize=6.0)
    ax.add_patch(plt.Rectangle((64, y0), 17, h0, facecolor="#ffffff", edgecolor="#555555", lw=0.6, hatch="////"))
    ax.text(72.5, y0 + h0 + 2.2, "DeepSpeed logging", ha="center", va="bottom", fontsize=5.8)
    for xm, lab in ((28, "call issued"), (81, "device sync returns")):
        ax.plot([xm, xm], [y0 - 4.5, y0 + h0 + 1.5], color="#222222", lw=0.8, ls="--")
        ax.text(xm, y0 + h0 + 6.0 if xm == 28 else y0 + h0 + 7.8, lab, ha="center", va="bottom", fontsize=5.8)
    ax.plot([28, 28, 81, 81], [y0 - 3.0, y0 - 5.5, y0 - 5.5, y0 - 3.0], color="#222222", lw=0.8)
    ax.text(54.5, y0 - 9.0, "timed interval", ha="center", va="center", fontsize=6.3)
    ax.text(50, 6.8, "overlap disabled; step time = maximum across ranks", ha="center", va="center", fontsize=6.0)
    ax.text(50, -1.5, "(a) state, observed collectives, and timer boundaries", ha="center", fontsize=7.8)
    box(bx, 1, 52, 47, 46, "Intel Max 1550", ["8 tiles per node", "oneCCL over TCP,", "host-staged"], "#e8eef5")
    box(bx, 52, 52, 47, 46, "NVIDIA H100", ["4 GPUs per node; NVLink", "within a node; NCCL over",
                                            "InfiniBand, host-staged;", "TCP-socket control"], "#e8eef5")
    box(bx, 1, 4, 98, 42, "Routing microbenchmark (two nodes)",
        ["one SwiGLU expert per rank, BF16", "dispatch all-to-all $\\rightarrow$ expert forward and backward",
         "$\\rightarrow$ return all-to-all $\\rightarrow$ final barrier wait",
         "balanced vs. half of every rank's tokens to expert 0"], "#fdf3d4", fs=6.5, step=7.6)
    bx.text(50, -1.5, "(b) configurations and routing phases", ha="center", fontsize=7.8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.06)
    fig.savefig(os.path.join(out, "fig_paths.pdf"), metadata=PDF_META)
    plt.close(fig)


def fig_zero(out, data, fd):
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.9))
    (a, b), (c, d) = axs
    # (a) DDP and ZeRO-2 median step time; their step time is stable
    fd["fig2a"] = []
    for p in ("pvc", "h100"):
        for m in ("ddp", "zero2"):
            ns, ys, es = [], [], []
            for n in (1, 2, 4):
                g = analyze.agg(data[p], n, m)
                ns.append(n)
                ys.append(g["step_med"] / 1e3)
                es.append(g["step_spread"] / 1e3)
            a.errorbar(ns, ys, yerr=es, color=COL[m], marker=MK[p], ls=LS[p], lw=1.1, ms=4, mfc="white",
                       capsize=2, label=f"{'Intel' if p == 'pvc' else 'H100'} {MODE[m]}")
            fd["fig2a"].append({"platform": p, "mode": m, "nodes": ns, "median_s": [round(y, 4) for y in ys],
                                "half_range_s": [round(e, 4) for e in es]})
    a.set_yscale("log")
    # the data span less than two decades, so label octaves rather than decades
    a.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    a.yaxis.set_minor_formatter(NullFormatter())
    a.set_ylim(0.2, 10)
    a.set_xticks([1, 2, 4])
    a.set_xlabel("nodes", fontsize=8)
    a.set_ylabel("median step (s)", fontsize=8)
    a.legend(fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False,
             labelspacing=0.25, handlelength=1.6, columnspacing=0.8)
    a.set_title("(a) DDP and ZeRO-2 step time", fontsize=8.5)
    # (b) peak memory per rank
    for p in ("pvc", "h100"):
        for m in ("ddp", "zero2", "zero3"):
            ns, ys = [], []
            for n in (1, 2, 4):
                g = analyze.agg(data[p], n, m)
                ns.append(n)
                ys.append(g["peak_gib"])
            b.plot(ns, ys, color=COL[m], marker=MK[p], ls=LS[p], lw=1.1, ms=4, mfc="white",
                   label=f"{'Intel' if p == 'pvc' else 'H100'} {MODE[m]}")
    b.set_xticks([1, 2, 4])
    b.set_xlabel("nodes", fontsize=8)
    b.set_ylabel("peak memory (GiB / rank)", fontsize=8)
    b.set_title("(b) peak memory", fontsize=8.5)
    b.legend(fontsize=5.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False,
             labelspacing=0.25, handlelength=1.6, columnspacing=0.8)
    # (c) four-node composition
    cells = [(p, m) for p in ("pvc", "h100") for m in ("ddp", "zero2", "zero3")]
    cats = [("compute", "#5b8db8", "compute + unattributed"), ("grad", "#7fcdbb", "gradient reduction"),
            ("gather", "#f5d76e", "parameter all-gather"), ("sync", "#f2b27e", "synchronization"),
            ("othercomm", "#c8c8c8", "other comm")]
    fd["fig2c"] = []
    for i, (p, m) in enumerate(cells):
        g = analyze.agg(data[p], 4, m)
        tot = sum(g[k] for k, _, _ in cats)
        bot = 0.0
        shares = {}
        for k, col, lab in cats:
            v = 100.0 * g[k] / tot
            shares[k] = round(v, 3)
            c.bar(i, v, 0.62, bottom=bot, color=col, edgecolor="#333333", linewidth=0.5,
                  label=lab if i == 0 else None)
            bot += v
        fd["fig2c"].append({"platform": p, "mode": m, "shares_percent": shares, "decomposed_file": g["decomposed_file"]})
    c.set_xticks(range(6))
    c.set_xticklabels(["DDP", "Z2", "Z3", "DDP", "Z2", "Z3"], fontsize=7.5)
    c.text(0.25, -0.24, "Intel", ha="center", fontsize=8, transform=c.transAxes)
    c.text(0.75, -0.24, "H100", ha="center", fontsize=8, transform=c.transAxes)
    c.axvline(2.5, color="#888888", lw=0.7, ls=":")
    c.set_ylabel("share of exposed step (%)", fontsize=8)
    c.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False, columnspacing=0.9)
    c.set_title("(c) four-node composition", fontsize=8.5)
    # (d) ZeRO-3 step time at each logged iteration
    traj = analyze.zero3_trajectories(PROCESSED)
    seen = set()
    fd["fig2d"] = []
    for r in sorted(traj, key=lambda r: (r["platform"], int(r["nodes"]), r["transport"], r["rep"])):
        w = int(r["warmup_iters"])
        pts = [(w + int(x), int(y) / 1e3) for x, y in (t.split(":") for t in r["heartbeats_ms"].split(";"))]
        plat, n = r["platform"], int(r["nodes"])
        tcp, host = r["transport"] == "sock", r["transport"] == "host"
        key = (plat, n, tcp, host)
        lab = None
        if key not in seen:
            seen.add(key)
            lab = "%s %d node%s%s" % ("H100" if plat == "h100" else "Intel", n, "" if n == 1 else "s",
                                      ", TCP" if tcp else (", host-staged" if host else ""))
        d.plot([q[0] for q in pts], [q[1] for q in pts], ls="--" if tcp else (":" if host else "-"),
               marker=MK[plat], ms=2.8, lw=0.9, color=analyze.TRAJ[(plat, n)],
               mfc="white" if tcp else analyze.TRAJ[(plat, n)], label=lab)
        fd["fig2d"].append({"platform": plat, "nodes": n, "transport": r["transport"], "rep": r["rep"],
                            "job": r["job"], "warmup": w, "measured": int(r["measured_iters"]),
                            "points_step_s": [[q[0], q[1]] for q in pts]})
    d.axvline(11, color="#888888", lw=0.6, ls=":")
    d.set_xlim(0, 44)
    d.set_ylim(0, 29)
    d.set_xlabel("training step (warm-up + measured)", fontsize=8)
    d.set_ylabel("ZeRO-3 step time (s)", fontsize=8)
    d.set_title("(d) ZeRO-3 step time per logged iteration", fontsize=8.5)
    d.legend(fontsize=5.4, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False,
             labelspacing=0.2, handlelength=1.8, columnspacing=0.8)
    for ax in (a, b, c, d):
        ax.tick_params(labelsize=7.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.35, color="#dddddd")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.95, bottom=0.18, hspace=0.95, wspace=0.28)
    fig.savefig(os.path.join(out, "fig_zero_expanded.pdf"), metadata=PDF_META)
    plt.close(fig)


def moe_cell(moe, p, mode):
    """Median across reps of the critical rank's phase p50s plus the measured
    step p50/p95."""
    runs = [r for r in moe[p] if r["mode"] == mode]
    phases = ["dispatch_ms", "expert_ms", "return_ms", "end_wait_ms"]
    world = runs[0]["world"]
    per_rank = {k: [st.median([r["per_rank_p50"][k][i] for r in runs]) for i in range(world)] for k in phases}
    totals = [sum(per_rank[k][i] for k in phases) for i in range(world)]
    crit = max(range(world), key=lambda i: totals[i])
    p50 = st.median([r["step_max_ms"]["p50"] for r in runs])
    p95 = st.median([r["step_max_ms"]["p95"] for r in runs])
    return {k: per_rank[k][crit] for k in phases}, p50, p95, per_rank, crit


def fig_moe(out, moe, fd):
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    a, b, c = axs
    cats = [("dispatch_ms", "#f5d76e", "dispatch all-to-all"), ("expert_ms", "#5b8db8", "expert computation"),
            ("return_ms", "#e0a838", "return all-to-all"), ("end_wait_ms", "#f2b27e", "final barrier wait")]
    fd["fig3"] = []
    for ax, p, title in ((a, "pvc", "Intel (16 ranks)"), (b, "h100", "H100 (8 ranks)")):
        for i, mode in enumerate(("balanced", "skewed")):
            ph, p50, p95, _, crit = moe_cell(moe, p, mode)
            bot = 0.0
            for k, col, lab in cats:
                ax.bar(i, ph[k], 0.55, bottom=bot, color=col, edgecolor="#333333", linewidth=0.5,
                       label=lab if (i == 0 and p == "pvc") else None)
                bot += ph[k]
            ax.plot([i - 0.33, i + 0.33], [p50, p50], color="#222222", lw=1.1)
            ax.errorbar(i, p50, yerr=[[0], [max(p95 - p50, 0)]], color="#222222", capsize=4, lw=1.0)
            ax.text(i + 0.3, p95, f" p95 {p95:.0f}", fontsize=6.6, va="center")
            fd["fig3"].append({"platform": p, "mode": mode, "critical_rank": crit,
                               **{k: round(v, 4) for k, v in ph.items()}, "p50_ms": p50, "p95_ms": p95})
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["balanced", "skewed"], fontsize=7.5)
        ax.set_ylabel("ms per iteration", fontsize=8)
        ax.set_title(title, fontsize=8.5)
    a.legend(fontsize=6.0, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86), labelspacing=0.25)
    for p, col, lab in (("pvc", "#0072B2", "Intel skewed"), ("h100", "#D55E00", "H100 skewed")):
        _, _, _, per_rank, _ = moe_cell(moe, p, "skewed")
        ex = per_rank["expert_ms"]
        med = st.median(ex)
        c.plot(range(len(ex)), [v / med for v in ex], marker="o", ms=3.2, lw=0.9, color=col, label=lab)
    c.axhline(1.0, color="#999999", lw=0.7, ls=":")
    c.set_xlabel("rank", fontsize=8)
    c.set_ylabel("expert time / median rank", fontsize=8)
    c.legend(fontsize=6.4, frameon=False)
    c.set_title("expert-time imbalance", fontsize=8.5)
    for ax in axs:
        ax.tick_params(labelsize=7.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.35, color="#dddddd")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.15, wspace=0.38)
    fig.savefig(os.path.join(out, "fig_moe_expanded.pdf"), metadata=PDF_META)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "figures", "companion"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    data = {p: analyze.llama_runs(analyze.load(p)) for p in ("pvc", "h100")}
    moe = {p: analyze.moe_runs(analyze.load(p)) for p in ("pvc", "h100")}
    fd = {}
    fig_paths(args.out)
    fig_zero(args.out, data, fd)
    fig_moe(args.out, moe, fd)
    with open(os.path.join(PROCESSED, "companion_figure_data.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(fd, fh, indent=1, sort_keys=True)
    print("wrote", ", ".join(sorted(os.listdir(args.out))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
