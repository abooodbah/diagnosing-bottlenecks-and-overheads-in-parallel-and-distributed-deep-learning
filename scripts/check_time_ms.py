# -*- coding: utf-8 -*-
"""Check that the invalid 'time_ms' fields of the 6 Sep records feed no
published value.

The harness of 6 Sep (lb-v0 to lb-v3) copied DeepSpeed's comms-logger
records into comms_per_rank0 and comms_all_ranks, reading the wrong list:
DeepSpeed 0.19.4 stores [count, latencies, algbw, busbw] and the harness
summed index 2, so 'time_ms' in those records is a sum of bandwidth values.
Records written that way are recognised by their layout (common.comm_layout).

The check reruns every script that reads the records (analyze.py with
figures, table3.py, logger_model.py) on copies of raw/ in which those fields
are multiplied by 1e6, set to zero, or negated, and requires every output
to be byte-identical to a run on an unaltered copy. As a control, the same
alteration of a record with valid times must change the outputs. It also
reports how the classification rule used until 2026-09-23, which inferred
the layout from the size of the times, would have treated the altered
records.

Writes processed/time_ms_check.txt. Exit status 1 if any check fails.
"""
import copy
import filecmp
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

from common import PROCESSED, RAW, ROOT, comm_layout

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = ["summary.csv", "figure_data.json", "table3_zero3_ops.csv", "table3_all_reps.csv",
           "zero3_layer_totals.csv", "logger_model.csv"]
FIGS = ["zero_migration.png", "transport_contrast.png", "moe_imbalance.png"]


def old_is_v2(run):
    """The rule in analyze.py until 2026-09-23 (kept here for comparison)."""
    if run.get("bench_version") == 2:
        return True
    if run.get("iters") != 40:
        return True
    ranks = [c for c in (run.get("comms_all_ranks") or []) if c]
    if not ranks:
        return False
    per_rank = (sum(sum(v["time_ms"] for v in c.values()) for c in ranks) / len(ranks) / run["iters"])
    return per_rank / run["step_max_ms"]["p50"] > 0.10


def alter(rec, how):
    r = copy.deepcopy(rec)
    for c in [r.get("comms_per_rank0")] + list(r.get("comms_all_ranks") or []):
        for v in (c or {}).values():
            v["time_ms"] = {"x1e6": v["time_ms"] * 1e6, "zero": 0.0, "neg": -v["time_ms"] - 1.0}[how]
    return r


def run_pipeline(raw_dir, out_dir):
    os.makedirs(os.path.join(out_dir, "processed"), exist_ok=True)
    for f in ("launch_windows.csv", "drift_launches.csv"):
        shutil.copy2(os.path.join(PROCESSED, f), os.path.join(out_dir, "processed", f))
    env = dict(os.environ, ZMIG_RAW_DIR=raw_dir, ZMIG_PROCESSED_DIR=os.path.join(out_dir, "processed"),
               PYTHONDONTWRITEBYTECODE="1")
    for cmd in (["analyze.py", "--raw", raw_dir, "--out", out_dir], ["table3.py"], ["logger_model.py"]):
        subprocess.run([sys.executable] + cmd, cwd=HERE, env=env, check=True, capture_output=True)


def same_outputs(a, b):
    diffs = []
    for f in OUTPUTS:
        if not filecmp.cmp(os.path.join(a, "processed", f), os.path.join(b, "processed", f), shallow=False):
            diffs.append(f)
    for f in FIGS:
        if not filecmp.cmp(os.path.join(a, "figures", f), os.path.join(b, "figures", f), shallow=False):
            diffs.append(f)
    return diffs


def main():
    lines, bad = [], 0
    invalid, valid_example = [], None
    for d in ("h100", "pvc", "pvc_v1"):
        for p in sorted(glob.glob(os.path.join(RAW, d, "*.json"))):
            rec = json.load(open(p))
            lay = comm_layout(rec)
            if lay == "deepspeed-logger":
                invalid.append((d, os.path.basename(p), rec))
            elif lay == "harness-timers" and valid_example is None and d == "pvc" and rec.get("mode") == "zero3":
                valid_example = (d, os.path.basename(p), rec)
    lines.append("Records whose comm fields are DeepSpeed logger records (time_ms is not a time):")
    for d, f, rec in invalid:
        used = "not read by any script (raw/pvc_v1 is outside the analysis)" if d == "pvc_v1" else \
            "step times used; comm fields excluded by layout"
        lines.append("  raw/%s/%s job %s: %s" % (d, f, rec["env"]["job"], used))
    tmp = tempfile.mkdtemp(prefix="time_ms_check_")
    try:
        base_raw = os.path.join(tmp, "raw_base")
        shutil.copytree(RAW, base_raw)
        run_pipeline(base_raw, os.path.join(tmp, "out_base"))
        # the unaltered copy must reproduce the repository's own outputs
        for f in OUTPUTS:
            ok = filecmp.cmp(os.path.join(tmp, "out_base", "processed", f), os.path.join(PROCESSED, f), shallow=False)
            bad += not ok
            lines.append("%-4s unaltered copy reproduces processed/%s" % ("ok" if ok else "BAD", f))
        for how in ("x1e6", "zero", "neg"):
            raw = os.path.join(tmp, "raw_" + how)
            shutil.copytree(RAW, raw)
            for d, f, rec in invalid:
                json.dump(alter(rec, how), open(os.path.join(raw, d, f), "w"), indent=1)
            out = os.path.join(tmp, "out_" + how)
            run_pipeline(raw, out)
            diffs = same_outputs(os.path.join(tmp, "out_base"), out)
            bad += bool(diffs)
            lines.append("%-4s invalid time_ms altered (%s): %s" % (
                "ok" if not diffs else "BAD", how, "all outputs byte-identical" if not diffs else "changed: " + ", ".join(diffs)))
            olds = [old_is_v2(alter(rec, how)) for d, f, rec in invalid if d != "pvc_v1"]
            lines.append("     the pre-2026-09-23 rule would treat %d of %d altered records as having valid times"
                         % (sum(olds), len(olds)))
        # control: altering valid times must change the outputs
        d, f, rec = valid_example
        raw = os.path.join(tmp, "raw_control")
        shutil.copytree(RAW, raw)
        json.dump(alter(rec, "x1e6"), open(os.path.join(raw, d, f), "w"), indent=1)
        out = os.path.join(tmp, "out_control")
        run_pipeline(raw, out)
        diffs = same_outputs(os.path.join(tmp, "out_base"), out)
        bad += not diffs
        lines.append("%-4s control: altering valid times in raw/%s/%s changes %s" % (
            "ok" if diffs else "BAD", d, f, ", ".join(diffs) if diffs else "nothing"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    lines.append("RESULT: %s" % ("all checks passed" if not bad else "%d checks FAILED" % bad))
    with open(os.path.join(PROCESSED, "time_ms_check.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
