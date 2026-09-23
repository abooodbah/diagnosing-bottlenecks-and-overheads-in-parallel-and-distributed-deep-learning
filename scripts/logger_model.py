# -*- coding: utf-8 -*-
"""Cost model of DeepSpeed's communication logger, fitted to the growth.

In DeepSpeed 0.19.4, comm.timed_op calls timers(name).elapsed(reset=False)
after every logged call, and utils/timer.py sums the elapsed time of every
event pair recorded for that name so far, synchronizing on each pair. A call
therefore costs c times the number of earlier calls of the same name, and a
step with k_n calls of name n grows by c * sum(k_n^2) over the previous step.

Calls per step come from the kept result records timed by the harness (the
DeepSpeed-layer names allgather_fn and reduce_scatter_fn are logged by
DeepSpeed under all_gather_into_tensor and reduce_scatter_tensor; the
harness's own per-step barrier is a torch.distributed call DeepSpeed does not
log). c is fitted per platform from the ZeRO-3 growth, and the same c
predicts the DDP and ZeRO-2 growth. Growth rates come from drift.py.

Output: processed/logger_model.csv
"""
import collections
import csv
import os
import statistics as st
import sys

from common import PROCESSED, comm_layout, load_records

LOG_NAME = {"allgather_fn": "all_gather_into_tensor", "reduce_scatter_fn": "reduce_scatter_tensor"}
NOT_LOGGED = {"barrier"}


def calls_per_step(rec):
    c0 = rec.get("comms_per_rank0") or {}
    out = collections.Counter()
    for name, v in c0.items():
        if name in NOT_LOGGED:
            continue
        out[LOG_NAME.get(name, name)] += v["count"] / rec["iters"]
    return {k: round(v, 3) for k, v in out.items() if v}


def main():
    drift = list(csv.DictReader(open(os.path.join(PROCESSED, "drift_launches.csv"), encoding="utf-8")))
    rows = []
    for plat in ("h100", "pvc"):
        per_mode = {}
        for rec in load_records(plat):
            if rec.get("mode") not in ("ddp", "zero2", "zero3") or comm_layout(rec) != "harness-timers":
                continue
            per_mode.setdefault(rec["mode"], collections.Counter())[tuple(sorted(calls_per_step(rec).items()))] += 1
        calls = {m: dict(cnt.most_common(1)[0][0]) for m, cnt in per_mode.items()}
        variants = {m: len(cnt) for m, cnt in per_mode.items()}
        k2 = {m: sum(v * v for v in c.values()) for m, c in calls.items()}
        kept = [r for r in drift if r["platform"] == plat and r["result_status"].startswith("kept in raw/")]
        z3 = [float(r["slope_all_ms"]) for r in kept if r["mode"] == "zero3"]
        c_fit = [s / k2["zero3"] for s in z3]              # ms per earlier call
        c_med = st.median(c_fit)
        for m in ("ddp", "zero2", "zero3"):
            sel = [r for r in kept if r["mode"] == m]
            # only the replicated baseline's first measured step runs slow
            key = "slope_after1_ms" if m == "ddp" else "slope_all_ms"
            meas = [float(r[key]) for r in sel if r[key] != ""]
            one = [float(r[key]) for r in sel if r[key] != "" and r["nodes"] == "1"]
            rows.append({
                "platform": plat, "mode": m,
                "logged_calls_per_step": ";".join("%s=%g" % kv for kv in sorted(calls[m].items())),
                "record_variants": variants[m], "sum_k_squared": round(k2[m], 1),
                "c_fit_us_median": round(1000 * c_med, 2),
                "c_fit_us_range": "%.2f-%.2f" % (1000 * min(c_fit), 1000 * max(c_fit)),
                "predicted_growth_ms_per_step": round(c_med * k2[m], 2),
                "measured_growth_statistic": ("least squares without the first measured iteration" if m == "ddp"
                                              else "least squares over all heartbeats"),
                "measured_growth_ms_median": round(st.median(meas), 2) if meas else "",
                "measured_growth_ms_min": round(min(meas), 2) if meas else "",
                "measured_growth_ms_max": round(max(meas), 2) if meas else "",
                "measured_one_node_min": round(min(one), 2) if one else "",
                "measured_one_node_max": round(max(one), 2) if one else "",
                "runs": len(meas),
            })
    with open(os.path.join(PROCESSED, "logger_model.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["platform"], r["mode"].ljust(5), "calls", r["logged_calls_per_step"].ljust(62), "sumk2", r["sum_k_squared"],
              "| c", r["c_fit_us_median"], "us (", r["c_fit_us_range"], ") | predicted", r["predicted_growth_ms_per_step"],
              "| measured", r["measured_growth_ms_median"], "[", r["measured_growth_ms_min"], "-", r["measured_growth_ms_max"], "] n", r["runs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
