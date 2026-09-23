# -*- coding: utf-8 -*-
"""Per-iteration collective accounting of the H100 ZeRO-3 runs timed at both
layers (companion report, Table III), and the timer-layer totals of every
ZeRO-3 record.

Table III: rank 0, one repetition per node count, the DeepSpeed-layer-timed
repetition with the shortest median step (1 node: rep 2 of job 3481929; 2
nodes: rep 3 of job 3482991, the only such run; 4 nodes: rep 1 of job
3476715). Per-iteration values are the window totals divided by the number
of measured iterations; the measured step is the record's median step (the
maximum across ranks). processed/table3_all_reps.csv gives every eligible
repetition, so the choice of repetition can be checked.

Layer totals (processed/zero3_layer_totals.csv), for every kept ZeRO-3
record: the timer layer, gathers, reduce-scatters and all timed collectives
per step for rank 0 and averaged over ranks, and the gather share of the
median step.

Inputs: raw/*/*.json, processed/launch_windows.csv.
"""
import csv
import os
import statistics as st
import sys

from common import PROCESSED, has_valid_comm_times, load_records, timed_at_deepspeed_layer

GATHER = ("allgather_fn", "all_gather_into_tensor", "all_gather", "allgather_base", "all_gather_base")
RS = ("reduce_scatter_fn", "reduce_scatter_tensor", "reduce_scatter")


def per_step(c, names, iters):
    return sum(v["time_ms"] for k, v in c.items() if k in names) / iters


def main():
    lw = {(r["job"], r["result_file"]): r for r in
          csv.DictReader(open(os.path.join(PROCESSED, "launch_windows.csv"), encoding="utf-8"))}
    recs = [r for p in ("h100", "pvc") for r in load_records(p) if r.get("mode") == "zero3"]

    # ---- Table III
    both = [r for r in recs if r["_dir"] == "h100" and timed_at_deepspeed_layer(r)]
    all_rows, table = [], []
    for n in (1, 2, 4):
        cand = sorted([r for r in both if r["nodes"] == n], key=lambda r: r["step_max_ms"]["p50"])
        for r in cand:
            c = r["comms_per_rank0"]
            it = r["iters"]
            row = {"nodes": n, "rep": r["rep"], "job": r["env"]["job"], "file": r["_file"],
                   "allgather_fn_calls": c["allgather_fn"]["count"] / it,
                   "allgather_fn_ms": round(c["allgather_fn"]["time_ms"] / it, 1),
                   "other_allgather_calls": c["all_gather_into_tensor"]["count"] / it,
                   "other_allgather_ms": round(c["all_gather_into_tensor"]["time_ms"] / it, 1),
                   "reduce_scatter_calls": sum(v["count"] for k, v in c.items() if k in RS) / it,
                   "reduce_scatter_ms": round(per_step(c, RS, it), 1),
                   "all_reduce_calls": c["all_reduce"]["count"] / it,
                   "all_reduce_ms": round(c["all_reduce"]["time_ms"] / it, 1),
                   "barrier_calls": c["barrier"]["count"] / it,
                   "barrier_ms": round(c["barrier"]["time_ms"] / it, 1),
                   "measured_step_ms": round(r["step_max_ms"]["p50"], 0)}
            ops = sum(v["time_ms"] for v in c.values()) / it
            gathers = per_step(c, GATHER, it)
            row["timed_ops_share"] = round(100 * ops / r["step_max_ms"]["p50"], 1)
            row["gathers_share"] = round(100 * gathers / r["step_max_ms"]["p50"], 1)
            all_rows.append(row)
        if cand:
            table.append(next(x for x in all_rows if x["nodes"] == n and x["file"] == cand[0]["_file"]))
    for name, rows in (("table3_zero3_ops.csv", table), ("table3_all_reps.csv", all_rows)):
        with open(os.path.join(PROCESSED, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    # ---- timer-layer totals of every ZeRO-3 record
    lt = []
    for r in sorted(recs, key=lambda r: (r["_dir"], r["nodes"], r["_file"])):
        it = r["iters"]
        if not has_valid_comm_times(r):
            # 6 Sep harness: comm fields are DeepSpeed logger records whose
            # time_ms is a bandwidth sum, not a time; no comm value is used
            lt.append({"platform": r["_dir"], "nodes": r["nodes"], "file": r["_file"], "job": r["env"]["job"],
                       "harness": lw.get((r["env"]["job"], r["_file"]), {}).get("harness", ""),
                       "window": "%s+%s" % (r.get("warmup"), it),
                       "timer_layer": "none (logger records; time_ms invalid)",
                       "transport": r["env"].get("transport_note", ""),
                       "step_p50_ms": round(r["step_max_ms"]["p50"], 1)})
            continue
        c0 = r["comms_per_rank0"]
        ranks = [c for c in r["comms_all_ranks"] if c]
        info = lw.get((r["env"]["job"], r["_file"]), {})
        lt.append({
            "platform": r["_dir"], "nodes": r["nodes"], "file": r["_file"], "job": r["env"]["job"],
            "harness": info.get("harness", ""), "window": "%s+%s" % (r.get("warmup"), it),
            "timer_layer": "DeepSpeed and torch" if timed_at_deepspeed_layer(r) else "torch only",
            "transport": r["env"].get("transport_note", ""),
            "step_p50_ms": round(r["step_max_ms"]["p50"], 1),
            "gathers_per_step_rank0_ms": round(per_step(c0, GATHER, it), 1),
            "gathers_per_step_mean_ms": round(st.mean(per_step(c, GATHER, it) for c in ranks), 1),
            "reduce_scatter_per_step_rank0_ms": round(per_step(c0, RS, it), 1),
            "collectives_per_step_rank0_ms": round(sum(v["time_ms"] for v in c0.values()) / it, 1),
            "collectives_per_step_mean_ms": round(st.mean(sum(v["time_ms"] for v in c.values()) / it for c in ranks), 1),
            "gather_share_of_p50_mean": round(st.mean(per_step(c, GATHER, it) for c in ranks) / r["step_max_ms"]["p50"], 3),
        })
    with open(os.path.join(PROCESSED, "zero3_layer_totals.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=max((list(x) for x in lt), key=len), lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(lt)
    for t in table:
        print("Table III n%d rep%s job %s: fn %.1f other %.1f rs %.1f ar %.1f bar %.1f step %d | ops %.1f%% gathers %.1f%%" % (
            t["nodes"], t["rep"], t["job"], t["allgather_fn_ms"], t["other_allgather_ms"], t["reduce_scatter_ms"],
            t["all_reduce_ms"], t["barrier_ms"], t["measured_step_ms"], t["timed_ops_share"], t["gathers_share"]))
    for x in lt:
        if "gathers_per_step_rank0_ms" not in x:
            print("%-4s n%d %-34s %s step %.1f" % (x["platform"], x["nodes"], x["file"], x["timer_layer"], x["step_p50_ms"]))
            continue
        print("%-4s n%d %-34s %-16s %-19s step %8.1f | gathers r0 %7.1f mean %7.1f | all coll r0 %7.1f mean %7.1f | share %.3f" % (
            x["platform"], x["nodes"], x["file"], x["transport"][:16], x["timer_layer"], x["step_p50_ms"],
            x["gathers_per_step_rank0_ms"], x["gathers_per_step_mean_ms"], x["collectives_per_step_rank0_ms"],
            x["collectives_per_step_mean_ms"], x["gather_share_of_p50_mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
