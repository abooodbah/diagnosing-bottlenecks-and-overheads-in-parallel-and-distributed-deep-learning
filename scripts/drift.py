# -*- coding: utf-8 -*-
"""Step-time drift in every training launch, from the heartbeat lines.

The harness prints rank 0's step time every tenth measured iteration
("iter m/N last=Xms"; m = 1, 11, 21, ...). A launch runs w warm-up steps and
then N measured steps, so measured iteration m is training step k = w + m.
The DeepSpeed communication logger's cost grows with the number of calls
logged since the engine started, so step times are compared at equal k, not
at equal m, and a window's median depends on w and N.

Per launch (processed/drift_launches.csv):
  slope_all_ms     least-squares growth per step over all heartbeats
  slope_after1_ms  the same, without the first measured iteration (the first
                   measured DDP and ZeRO-2 steps run slower)
  at_step11_ms     step time at training step 11 (the first measured step of
                   a 10-warm-up run), interpolated between heartbeats
  at_step27_ms     step time at step 27, the last step logged in every run
  growth_share     (window median - first measured step) / window median,
                   for launches whose record was kept
Per published cell (processed/drift_cells.csv): the windows the cell mixes,
the published median, and the matched-step values and slopes of its runs.
No value is extrapolated beyond the logged steps.
"""
import csv
import os
import statistics as st
import sys

from common import PROCESSED, interpolate, least_squares_slope

K_EARLY, K_LATE = 11, 27


def fnum(x, nd=1):
    return "" if x is None else round(x, nd)


def main():
    rows = list(csv.DictReader(open(os.path.join(PROCESSED, "launch_windows.csv"), encoding="utf-8")))
    out = []
    for r in rows:
        if r["program"] != "llama" or not r["heartbeats_ms"]:
            continue
        w = int(r["warmup_iters"])
        beats = [(int(a), int(b)) for a, b in (x.split(":") for x in r["heartbeats_ms"].split(";"))]
        pts = [(w + m, v) for m, v in beats]
        s_all = least_squares_slope(pts)
        s_after = least_squares_slope(pts[1:]) if len(pts) >= 3 else None
        p50 = float(r["step_p50_ms"]) if r["step_p50_ms"] else None
        kept = r["result_status"].startswith("kept in raw/")
        share = (p50 - beats[0][1]) / p50 if (p50 and kept) else None
        mode = {"stage=0": "ddp", "stage=2": "zero2", "stage=3": "zero3"}[r["stage_or_mode"]]
        out.append({
            "job": r["job"], "launch_time_cdt": r["launch_time_cdt"], "platform": r["platform"],
            "nodes": r["nodes"], "mode": mode, "rep": r["rep"], "transport": r["transport"],
            "harness": r["harness"], "result_status": r["result_status"], "result_file": r["result_file"],
            "warmup_iters": w, "measured_iters": r["measured_iters"],
            "first_logged_step": pts[0][0], "last_logged_step": pts[-1][0],
            "first_measured_ms": beats[0][1],
            "slope_all_ms": fnum(s_all, 2), "slope_after1_ms": fnum(s_after, 2),
            "at_step11_ms": fnum(interpolate(pts, K_EARLY)), "at_step27_ms": fnum(interpolate(pts, K_LATE)),
            "window_median_ms": p50 if p50 is not None else "",
            "growth_share": fnum(share, 3),
            "heartbeats_ms": r["heartbeats_ms"],
        })
    with open(os.path.join(PROCESSED, "drift_launches.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(out)

    # published cells: kept records only, the same selection as analyze.agg
    cells = {}
    for r in out:
        if not r["result_status"].startswith("kept in raw/"):
            continue
        tr = r["transport"]
        variant = "tcp-sockets" if tr == "sock" else ("host-staged" if tr == "host" else "default")
        cells.setdefault((r["platform"], int(r["nodes"]), r["mode"], variant), []).append(r)
    crow = []
    for key in sorted(cells, key=lambda k: (k[0], k[2], k[3], k[1])):
        rs = sorted(cells[key], key=lambda r: int(r["rep"]))
        windows = sorted({"%s+%s" % (r["warmup_iters"], r["measured_iters"]) for r in rs})
        med = st.median([float(r["window_median_ms"]) for r in rs])
        e11 = [float(r["at_step11_ms"]) for r in rs if r["at_step11_ms"] != ""]
        e27 = [float(r["at_step27_ms"]) for r in rs if r["at_step27_ms"] != ""]
        sl = [float(r["slope_all_ms"]) for r in rs if r["slope_all_ms"] != ""]
        sl1 = [float(r["slope_after1_ms"]) for r in rs if r["slope_after1_ms"] != ""]
        shares = [float(r["growth_share"]) for r in rs if r["growth_share"] != ""]
        med_rep = sorted(rs, key=lambda r: float(r["window_median_ms"]))[len(rs) // 2]
        crow.append({
            "platform": key[0], "nodes": key[1], "mode": key[2], "variant": key[3], "reps": len(rs),
            "windows_warmup+measured": ";".join(windows),
            "published_median_ms": round(med, 1),
            "step11_median_ms": fnum(st.median(e11)) if e11 else "", "step11_min_ms": fnum(min(e11)) if e11 else "",
            "step11_max_ms": fnum(max(e11)) if e11 else "",
            "step27_median_ms": fnum(st.median(e27)) if e27 else "",
            "slope_all_min_ms": fnum(min(sl), 2) if sl else "", "slope_all_max_ms": fnum(max(sl), 2) if sl else "",
            "slope_after1_median_ms": fnum(st.median(sl1), 2) if sl1 else "",
            "growth_share_min": fnum(min(shares), 3) if shares else "",
            "growth_share_max": fnum(max(shares), 3) if shares else "",
            "growth_share_median_rep": fnum((float(med_rep["window_median_ms"]) - float(med_rep["first_measured_ms"]))
                                            / float(med_rep["window_median_ms"]), 3),
        })
    with open(os.path.join(PROCESSED, "drift_cells.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(crow[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(crow)
    print("launches with heartbeats %d | published cells %d" % (len(out), len(crow)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
