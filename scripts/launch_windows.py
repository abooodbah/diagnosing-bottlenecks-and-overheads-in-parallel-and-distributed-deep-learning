# -*- coding: utf-8 -*-
"""One row per benchmark launch in the campaign job logs.

For every launch this records the measurement window actually used
(warm-up and measured iterations, with the source of each value), the
harness, launch-script and rank-helper versions that were on Stampede3 at
that moment, what happened to the result file (kept, set aside, overwritten
by a later job, or never written), and the evidence that ties the launch to
its harness version:

  timeline     the version uploaded last before the launch started
               (provenance/version_timeline.csv);
  fingerprint  llama_bench.py line numbers printed in stack traces, which
               must fall on a call inside the named function;
  markers      fields of the result record that only some versions write
               (bench_version, timer layout, DeepSpeed-layer timers, the
               transport rename, the default model depth, heartbeats).

Inputs: transport_logs/job_logs/*.out, provenance/*.csv, provenance/versions/,
raw/*/*.json. Outputs: processed/launch_windows.csv, processed/result_files.csv.
"""
import ast
import csv
import os
import re
import sys
from datetime import datetime, timedelta

from common import (PROCESSED, ROOT, comm_layout, job_logs, load_records, parse_log,
                    seconds_from_nearest_upload, slurm_jobs, slurm_steps, submit_env,
                    timed_at_deepspeed_layer, version_live, version_timeline)


# ------------------------------------------------------------ version facts
def version_features(text):
    layers = re.search(r'"--layers", type=int, default=(\d+)', text)
    return {
        "bench_version": '"bench_version": 2' in text,
        "harness_timers": "def install_comm_timers" in text,
        "deepspeed_layer_timers": "def install_ds_comm_timers" in text,
        "renames_direct": '"ib-gdr-disabled"' in text,
        "heartbeats": 'log0(f"iter' in text,
        "default_layers": int(layers.group(1)) if layers else None,
    }


def function_map(text):
    tree = ast.parse(text)
    owner = ["<module>"] * (text.count("\n") + 2)
    spans = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    for a, b, name in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
        for i in range(a, b + 1):
            owner[i] = name
    return owner, text.split("\n")


def fits_frames(fmap, frames):
    owner, lines = fmap
    return all(ln < len(owner) and owner[ln] == fn and "(" in lines[ln - 1] for ln, fn in frames)


def marker_versions(rec, launch, llama_versions):
    """Harness versions whose code could have written this record."""
    ok = []
    for label, feat in llama_versions:
        c = True
        if rec is not None:
            c &= feat["bench_version"] == (rec.get("bench_version") == 2)
            c &= feat["harness_timers"] == (comm_layout(rec) == "harness-timers")
            if rec.get("stage") == 3:
                c &= feat["deepspeed_layer_timers"] == timed_at_deepspeed_layer(rec)
            note = rec.get("env", {}).get("transport_note", "")
            if note == "ib-gdr-disabled":
                c &= feat["renames_direct"]
            elif note == "direct":
                c &= not feat["renames_direct"]
            if launch["params"].get("extra", "") != "--tiny":
                c &= feat["default_layers"] == rec.get("model", {}).get("layers")
        if launch["wrote"] or launch["heartbeats"]:
            c &= feat["heartbeats"] == bool(launch["heartbeats"])
        if c:
            ok.append(label)
    return ok


# ------------------------------------------------------------ windows
def script_window(script_text, env, program, tiny):
    """(warmup, measured) implied by the launch script, the submit-time
    environment and the harness defaults."""
    if program == "moe":
        return 5, 50, "harness default (routing benchmark)"
    if tiny:
        return 3, 8, "--tiny preset of the harness"
    it = re.search(r'--iters "\$\{ZMIG_ITERS:-(\d+)\}"', script_text)
    wu = re.search(r'--warmup "\$\{ZMIG_WARMUP:-(\d+)\}"', script_text)
    iters = int(env.get("ZMIG_ITERS", it.group(1))) if it else 60
    warm = int(env.get("ZMIG_WARMUP", wu.group(1))) if wu else 10
    src = "launch script and submit line" + ("" if wu else "; script passes no --warmup, harness default 10")
    return warm, iters, src


def main():
    timeline = version_timeline()
    jobs = slurm_jobs()
    steps = slurm_steps()
    logs = job_logs()

    llama_versions, fmaps = [], {}
    for r in timeline:
        if r["file"] == "llama_bench.py":
            text = open(os.path.join(ROOT, r["path"]), encoding="utf-8").read()
            llama_versions.append((r["label"], version_features(text)))
            fmaps[r["label"]] = function_map(text)
    script_text = {r["label"]: open(os.path.join(ROOT, r["path"]), encoding="utf-8").read()
                   for r in timeline if r["file"] in ("zmig_h100.slurm", "zmig_pvc.slurm")}

    records = {}
    for d in ("h100", "pvc", "pvc_v1"):
        for rec in load_records(d):
            records.setdefault((rec["_file"], rec.get("env", {}).get("job", "")), rec)

    rows = []
    for job, path in logs.items():
        J = jobs.get(job)
        if not J or J["start_dt"] is None:
            continue
        platform = "h100" if "h100" in J["job_name"] else ("pvc" if "pvc" in J["job_name"] else None)
        if platform is None:
            continue
        env = submit_env(J["submit_line"])
        sfile = "zmig_h100.slurm" if platform == "h100" else "zmig_pvc.slurm"
        sv = version_live(timeline, sfile, J["submit_dt"])
        job_steps = [s for s in steps if s["step_id"].split(".")[0] == job and s["start_dt"]]
        for L in parse_log(path):
            hh = datetime.strptime(L["hhmmss"], "%H:%M:%S").time()
            t = datetime.combine(J["start_dt"].date(), hh)
            if t < J["start_dt"] - timedelta(minutes=1):
                t += timedelta(days=1)
            st = min(job_steps, key=lambda s: abs((s["start_dt"] - t).total_seconds()), default=None)
            if st is not None and abs((st["start_dt"] - t).total_seconds()) <= 2:
                step_id, t = st["step_id"], st["start_dt"]
            else:
                step_id = ""
            prog = L["kind"]
            hfile = "llama_bench.py" if prog == "llama" else "moe_bench.py"
            hv = version_live(timeline, hfile, t)
            ev = version_live(timeline, "rank_env.sh", t)
            p = L["params"]
            tiny = p.get("extra", "") == "--tiny"
            name, log_p50 = (L["wrote"][0] if L["wrote"] else ("", None))
            rec = records.get((name, job)) if name else None
            if rec is not None:
                warm, meas = rec.get("warmup"), rec.get("iters")
                if warm is None and prog == "moe":
                    warm = 5
                wsrc = msrc = "result record" if rec.get("warmup") is not None else "result record (iters); harness default (warm-up)"
            else:
                warm, meas, src = script_window(script_text[sv["label"]], env, prog, tiny)
                wsrc = src
                msrc = src
                if L["heartbeats"]:
                    meas = L["heartbeats"][0][1]
                    msrc = "log heartbeat denominator"
            fp = ""
            if L["frames"] and prog == "llama":
                fits = [lab for lab, _ in llama_versions if fits_frames(fmaps[lab], L["frames"])]
                fp = ";".join(fits)
            mk = (";".join(marker_versions(rec, L, llama_versions))
                  if prog == "llama" and (rec or L["heartbeats"] or L["wrote"]) else "")
            agree = "n/a"
            if prog == "llama" and (fp or mk):
                ok = True
                if fp:
                    ok &= hv["label"] in fp.split(";")
                if mk:
                    ok &= hv["label"] in mk.split(";")
                agree = "yes" if ok else "NO"
            if prog == "llama":
                stage = "stage=" + p.get("stage", "?")
            else:
                stage = "mode=" + p.get("mode", "?")
            rows.append({
                "job": job, "slurm_step": step_id, "launch_time_cdt": t.isoformat(sep=" "),
                "platform": platform, "nodes": J["nodes"], "program": prog, "stage_or_mode": stage,
                "rep": p.get("rep", ""), "transport": p.get("transport", "tcp-hoststaged" if platform == "pvc" else ""),
                "tiny": "yes" if tiny else "",
                "warmup_iters": warm, "warmup_source": wsrc, "measured_iters": meas, "measured_source": msrc,
                "launch_script": sv["label"], "launch_script_category": sv["category"],
                "harness": hv["label"], "harness_category": hv["category"],
                "seconds_from_nearest_upload": round(seconds_from_nearest_upload(timeline, hfile, t)),
                "rank_env": ev["label"] if ev else "",
                "result_file": name, "result_status": "", "step_p50_ms": "", "p50_source": "",
                "rc": L["rc"] if L["rc"] is not None else "", "killed_at_time_limit": "yes" if L["killed"] else "",
                "heartbeats_ms": ";".join("%d:%d" % (m, v) for m, _, v in L["heartbeats"]),
                "fingerprint_fits": fp, "record_markers_fit": mk, "timeline_agrees": agree,
                "_rec": rec, "_log_p50": log_p50, "_t": t,
            })

    # result-file fate: writers of each name in time order; the last one survives
    writers = {}
    for r in rows:
        if r["result_file"]:
            writers.setdefault((r["platform"], r["result_file"]), []).append(r)
    fates = []
    for (plat, name), ws in sorted(writers.items()):
        ws.sort(key=lambda r: r["_t"])
        for i, r in enumerate(ws):
            rec = r["_rec"]
            if rec is not None and rec["_dir"] == "pvc_v1":
                r["result_status"] = "kept, set aside in raw/pvc_v1 (not used)"
            elif rec is not None:
                r["result_status"] = "kept in raw/%s" % rec["_dir"]
            else:
                later = [w for w in ws[i + 1:] if w["_rec"] is None or w["_rec"]["_dir"] != "pvc_v1"]
                r["result_status"] = ("overwritten by job %s" % later[0]["job"]) if later else "not in the artifact"
            if rec is not None:
                r["step_p50_ms"] = round(rec["step_max_ms"]["p50"], 1)
                r["p50_source"] = "result record"
            elif r["_log_p50"] is not None:
                r["step_p50_ms"] = r["_log_p50"]
                r["p50_source"] = "log WROTE line"
        survivor = next((w for w in reversed(ws) if w["_rec"] is not None), None)
        fates.append({"platform": plat, "result_file": name,
                      "writers": " ".join("%s@%s" % (w["job"], w["_t"].strftime("%m-%d %H:%M")) for w in ws),
                      "kept_from_job": survivor["job"] if survivor else "",
                      "location": ("raw/%s/%s" % (survivor["_rec"]["_dir"], name)) if survivor else "not in the artifact"})
    for r in rows:
        if not r["result_file"]:
            r["result_status"] = ("not written: killed at the time limit" if r["killed_at_time_limit"]
                                  else "not written: exit code %s" % r["rc"] if r["rc"] not in ("", 0)
                                  else "not written: job ended before the write")

    rows.sort(key=lambda r: r["_t"])
    os.makedirs(PROCESSED, exist_ok=True)
    cols = [k for k in rows[0] if not k.startswith("_")]
    with open(os.path.join(PROCESSED, "launch_windows.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(PROCESSED, "result_files.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fates[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(fates)
    bad = [r for r in rows if r["timeline_agrees"] == "NO"]
    print("launches %d | result names %d | timeline disagreements %d" % (len(rows), len(fates), len(bad)))
    for r in bad:
        print("  DISAGREE", r["job"], r["launch_time_cdt"], r["harness"], r["fingerprint_fits"], r["record_markers_fit"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
