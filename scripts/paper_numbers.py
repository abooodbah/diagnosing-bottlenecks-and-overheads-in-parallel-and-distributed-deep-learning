# -*- coding: utf-8 -*-
"""The sharding numbers quoted in the companion report and in Section 9 of
the survey, computed from the processed outputs, each with its definition,
so every quoted value can be checked against one file. The routing values
are in processed/companion_figure_data.json (companion Table IV and Fig. 3)
and processed/figure_data.json (survey Fig. 6).

Inputs: processed/summary.csv, drift_launches.csv, drift_cells.csv,
launch_windows.csv, logger_model.csv, table3_zero3_ops.csv,
zero3_layer_totals.csv, figure_data.json, companion_figure_data.json.
Output: processed/paper_numbers.csv
"""
import csv
import json
import os
import statistics as st
import sys

import analyze
from common import PROCESSED


def rows_of(name):
    return list(csv.DictReader(open(os.path.join(PROCESSED, name), encoding="utf-8")))


def main():
    out = []

    def put(key, value, unit, definition):
        out.append({"key": key, "value": value, "unit": unit, "definition": definition})

    summ = {(r["platform"], int(r["nodes"]), r["mode"]): r for r in rows_of("summary.csv")}
    cells = {(r["platform"], int(r["nodes"]), r["mode"], r["variant"]): r for r in rows_of("drift_cells.csv")}
    dl = [r for r in rows_of("drift_launches.csv") if r["result_status"].startswith("kept in raw/")]

    # ---- ZeRO-2 and DDP
    for p in ("h100", "pvc"):
        for n in (1, 2, 4):
            for m in ("ddp", "zero2", "zero3"):
                put("median_step_%s_n%d_%s" % (p, n, m), summ[(p, n, m)]["step_p50_ms"], "ms",
                    "median over repetitions of each run's median step (processed/summary.csv)")
        runs = analyze.llama_runs(analyze.load(p))       # unrounded medians, as the papers use
        z4, d4 = analyze.agg(runs, 4, "zero2")["step_med"], analyze.agg(runs, 4, "ddp")["step_med"]
        put("zero2_vs_ddp_4node_%s" % p, round(100 * (z4 - d4) / d4, 1), "%", "four-node ZeRO-2 median over DDP median")

    # ---- windows per published ZeRO-3 cell
    for (p, n, m, v), c in sorted(cells.items()):
        if m == "zero3":
            put("windows_%s_n%d_%s" % (p, n, v), c["windows_warmup+measured"], "warm-up+measured",
                "measurement windows of the repetitions in the cell (launch_windows.csv)")

    # ---- ZeRO-3 growth
    for p in ("h100", "pvc"):
        sl = [float(r["slope_all_ms"]) for r in dl if r["platform"] == p and r["mode"] == "zero3"]
        put("zero3_growth_%s_min" % p, round(min(sl) / 1e3, 3), "s per step", "least-squares slope over the heartbeats")
        put("zero3_growth_%s_max" % p, round(max(sl) / 1e3, 3), "s per step", "least-squares slope over the heartbeats")

    # ---- ZeRO-3 at matched training steps (median of the repetitions)
    for (p, n, m, v), c in sorted(cells.items()):
        if m != "zero3":
            continue
        put("zero3_step11_%s_n%d_%s" % (p, n, v), round(float(c["step11_median_ms"]) / 1e3, 2), "s",
            "step time at training step 11 (warm-up + measured), median of the repetitions; "
            "interpolated between logged iterations where step 11 was not logged")
        put("zero3_step27_%s_n%d_%s" % (p, n, v), round(float(c["step27_median_ms"]) / 1e3, 2), "s",
            "step time at training step 27, the last step logged in every run")
    h1 = [float(r["first_measured_ms"]) for r in dl if r["platform"] == "h100" and r["mode"] == "zero3" and r["nodes"] == "1"]
    put("zero3_h100_n1_first_measured_range", "%.1f-%.1f" % (min(h1) / 1e3, max(h1) / 1e3), "s", "first measured iteration")
    h31 = []
    for r in dl:
        if r["platform"] == "h100" and r["mode"] == "zero3" and r["nodes"] == "1":
            beats = dict((int(a), int(b)) for a, b in (x.split(":") for x in r["heartbeats_ms"].split(";")))
            h31.append(beats[31])
    put("zero3_h100_n1_iter31_range", "%.1f-%.1f" % (min(h31) / 1e3, max(h31) / 1e3), "s", "31st measured iteration")

    # ---- ratios to the replicated baseline at four nodes, at matched steps
    ddp4 = analyze.agg(analyze.llama_runs(analyze.load("h100")), 4, "ddp")["step_med"]
    z3_n4 = [r for r in dl if r["platform"] == "h100" and r["mode"] == "zero3" and r["nodes"] == "4"]
    at11 = st.median(float(r["at_step11_ms"]) for r in z3_n4)
    last = st.median(dict((int(a), int(b)) for a, b in (x.split(":") for x in r["heartbeats_ms"].split(";")))[31]
                     for r in z3_n4)
    put("h100_n4_zero3_over_ddp_step11", round(at11 / ddp4, 1), "x", "ZeRO-3 at step 11 over the DDP median")
    put("h100_n4_zero3_over_ddp_step41", round(last / ddp4, 1), "x", "ZeRO-3 at step 41 (31st measured) over the DDP median")
    put("h100_n4_zero3_over_ddp_median", round(float(summ[("h100", 4, "zero3")]["step_p50_ms"]) / ddp4, 1), "x",
        "40-iteration ZeRO-3 median over the DDP median")

    # ---- growth share (window median minus first measured step) / median
    for p in ("h100", "pvc"):
        cs = [c for (pp, n, m, v), c in cells.items() if pp == p and m == "zero3" and v == "default"]
        put("growth_share_median_rep_%s_min" % p, min(float(c["growth_share_median_rep"]) for c in cs), "fraction",
            "per cell, the median repetition's (median - first measured) / median")
        put("growth_share_median_rep_%s_max" % p, max(float(c["growth_share_median_rep"]) for c in cs), "fraction",
            "per cell, the median repetition's (median - first measured) / median")
        runs = [float(r["growth_share"]) for r in dl if r["platform"] == p and r["mode"] == "zero3" and r["growth_share"]]
        put("growth_share_runs_%s" % p, "%.3f-%.3f" % (min(runs), max(runs)), "fraction", "the same, over every run")

    # ---- transport contrast at matched steps
    fd = json.load(open(os.path.join(PROCESSED, "figure_data.json"), encoding="utf-8"))
    put("tcp_minus_ib_min", round(fd["transport_gap_ms"]["min"] / 1e3, 2), "s", "TCP minus InfiniBand, same rep and logged step")
    put("tcp_minus_ib_max", round(fd["transport_gap_ms"]["max"] / 1e3, 2), "s", "TCP minus InfiniBand, same rep and logged step")

    # ---- DDP and ZeRO-2 growth, logger model
    for r in rows_of("logger_model.csv"):
        put("logger_c_us_%s" % r["platform"], r["c_fit_us_median"], "us per earlier call", "fitted to the ZeRO-3 growth")
        put("predicted_growth_%s_%s" % (r["platform"], r["mode"]), r["predicted_growth_ms_per_step"], "ms per step",
            "c times the sum of squared calls per step")
        put("measured_growth_%s_%s" % (r["platform"], r["mode"]),
            "%s-%s" % (r["measured_growth_ms_min"], r["measured_growth_ms_max"]), "ms per step", r["measured_growth_statistic"])
        put("measured_growth_one_node_%s_%s" % (r["platform"], r["mode"]),
            "%s-%s" % (r["measured_one_node_min"], r["measured_one_node_max"]), "ms per step", "one-node runs only")

    # ---- Table III and timer layers
    for r in rows_of("table3_zero3_ops.csv"):
        put("table3_n%s" % r["nodes"], "fn %s other %s rs %s ar %s bar %s step %s ops %s%% gathers %s%%" % (
            r["allgather_fn_ms"], r["other_allgather_ms"], r["reduce_scatter_ms"], r["all_reduce_ms"],
            r["barrier_ms"], r["measured_step_ms"], r["timed_ops_share"], r["gathers_share"]), "ms", "rank 0 per iteration")
    lt = rows_of("zero3_layer_totals.csv")
    ib = [r for r in lt if r["platform"] == "h100" and r["nodes"] == "2" and r["timer_layer"] == "torch only"
          and r["transport"] != "tcp-sockets"]
    tcp = [r for r in lt if r["platform"] == "h100" and r["nodes"] == "2" and r["timer_layer"] == "torch only"
           and r["transport"] == "tcp-sockets"]
    for tag, rs in (("ib", ib), ("tcp", tcp)):
        g = [float(r["gathers_per_step_rank0_ms"]) for r in rs]
        a = [float(r["collectives_per_step_rank0_ms"]) for r in rs]
        s = [float(r["step_p50_ms"]) for r in rs]
        put("torchonly_gathers_%s" % tag, "%.1f-%.1f" % (min(g) / 1e3, max(g) / 1e3), "s per step", "rank 0")
        put("torchonly_collectives_%s" % tag, "%.2f-%.2f" % (min(a) / 1e3, max(a) / 1e3), "s per step", "rank 0")
        put("torchonly_steps_%s" % tag, "%.1f-%.1f" % (min(s) / 1e3, max(s) / 1e3), "s", "median step")
    pv = [float(r["gather_share_of_p50_mean"]) for r in lt if r["platform"] == "pvc" and r.get("gather_share_of_p50_mean")]
    put("intel_zero3_gather_share_runs", "%.3f-%.3f" % (min(pv), max(pv)), "fraction", "mean over ranks / median step, per run")
    cfd = json.load(open(os.path.join(PROCESSED, "companion_figure_data.json"), encoding="utf-8"))
    sh = [x["shares_percent"]["gather"] for x in cfd["fig2c"] if x["platform"] == "pvc" and x["mode"] == "zero3"]
    put("intel_zero3_gather_share_fig2c_4node", round(sh[0], 1), "%", "Fig. 2(c), four nodes")

    # ---- spread of repetitions
    for (p, n, m), r in sorted(summ.items()):
        put("half_range_pct_%s_n%d_%s" % (p, n, m), round(100 * float(r["spread_ms"]) / float(r["step_p50_ms"]), 1),
            "%", "half the range of the repetitions' medians, over the cell median")

    with open(os.path.join(PROCESSED, "paper_numbers.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "value", "unit", "definition"], lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    for o in out:
        print("%-44s %-60s %s" % (o["key"], o["value"], o["unit"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
