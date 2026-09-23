# -*- coding: utf-8 -*-
"""Shared readers for the raw run records, the Slurm job logs, the Slurm
accounting extract and the harness version timeline.

All paths are relative to the repository root, and nothing here writes.
"""
import csv
import glob
import hashlib
import json
import os
import re
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ZMIG_RAW_DIR and ZMIG_PROCESSED_DIR redirect inputs and outputs; the
# time_ms check uses them to rerun the pipeline on altered copies
RAW = os.environ.get("ZMIG_RAW_DIR") or os.path.join(ROOT, "raw")
LOGS = os.path.join(ROOT, "transport_logs", "job_logs")
PROV = os.path.join(ROOT, "provenance")
PROCESSED = os.environ.get("ZMIG_PROCESSED_DIR") or os.path.join(ROOT, "processed")
FIGURES = os.path.join(ROOT, "figures")

MODES = ("ddp", "zero2", "zero3")
STAGE_MODE = {0: "ddp", 2: "zero2", 3: "zero3"}
UTC_TO_CDT = timedelta(hours=-5)  # the campaign ran in September (CDT, UTC-5)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ------------------------------------------------------------ raw records
def load_records(platform, raw_dir=None):
    """All JSON records of one platform directory (raw/h100, raw/pvc, ...)."""
    runs = []
    for p in sorted(glob.glob(os.path.join(raw_dir or RAW, platform, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        d["_file"] = os.path.basename(p)
        d["_dir"] = platform
        runs.append(d)
    return runs


def comm_layout(rec):
    """How the record's communication fields were produced.

    'harness-timers'  : synchronized timers of the harness (lb-v4 and later);
                        every measured step times one barrier, so the rank-0
                        barrier count equals the number of measured steps, and
                        'time_ms' is milliseconds.
    'deepspeed-logger': the 6 Sep harness (lb-v0 to lb-v3) copied DeepSpeed's
                        comms-logger records. 'count' is valid, but 'time_ms'
                        is the sum of the logger's algorithmic-bandwidth list
                        (DeepSpeed 0.19.4 stores [count, latencies, algbw,
                        busbw]; the harness read index 2). It is not a time.
    None              : no communication fields (routing records).
    """
    c0 = rec.get("comms_per_rank0")
    ranks = rec.get("comms_all_ranks")
    if c0 is None and not ranks:
        return None
    barrier = (c0 or {}).get("barrier", {}).get("count")
    if rec.get("bench_version") == 2 or barrier == rec.get("iters"):
        return "harness-timers"
    return "deepspeed-logger"


def has_valid_comm_times(rec):
    return comm_layout(rec) == "harness-timers"


def timed_at_deepspeed_layer(rec):
    return any("allgather_fn" in (c or {}) for c in (rec.get("comms_all_ranks") or []))


# ------------------------------------------------------------ job logs
_HDR = re.compile(r"^--- (llama|moe) (.*?) (\d\d:\d\d:\d\d) ---")
_BEAT = re.compile(r"\[bench\] iter (\d+)/(\d+) last=(\d+)ms")
_WROTE = re.compile(r"\[(?:bench|moe)\] WROTE (\S+\.json) step_p50=([\d.]+)ms")
_RC = re.compile(r"^--- rc=(\d+) ---")
_TIMEOUT = re.compile(r"CANCELLED AT (\S+) DUE TO TIME LIMIT")
_FRAME_PY = re.compile(r'llama_bench\.py", line (\d+), in ([\w<>]+)')
_FRAME_NCCL = re.compile(r"#\d+ ([\w<>]+) from \S*llama_bench\.py:(\d+)")
_KV = re.compile(r"(\w+)=('[^']*'|\S+)")


def _params(text):
    out = {}
    for k, v in _KV.findall(text):
        out[k] = v.strip("'")
    return out


def parse_log(path):
    """Launch segments of one job log, in order.

    Each launch has: kind ('llama' or 'moe'), params (stage, rep, transport
    or extra, mode), hhmmss (printed just before srun), heartbeats
    [(measured_iteration, n_measured, rank0_ms)], wrote [(name, p50_ms)],
    rc, killed (time-limit cancellation), frames {(line, function)}."""
    launches = []
    cur = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _HDR.match(line)
            if m:
                cur = {"kind": m.group(1), "params": _params(m.group(2)), "hhmmss": m.group(3),
                       "heartbeats": [], "wrote": [], "rc": None, "killed": False, "frames": set()}
                launches.append(cur)
                continue
            if cur is None:
                continue
            m = _BEAT.search(line)
            if m:
                cur["heartbeats"].append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
                continue
            m = _WROTE.search(line)
            if m:
                cur["wrote"].append((m.group(1), float(m.group(2))))
                continue
            m = _RC.match(line)
            if m:
                cur["rc"] = int(m.group(1))
                continue
            if _TIMEOUT.search(line):
                cur["killed"] = True
            for m in _FRAME_PY.finditer(line):
                cur["frames"].add((int(m.group(1)), m.group(2)))
            for m in _FRAME_NCCL.finditer(line):
                cur["frames"].add((int(m.group(2)), m.group(1)))
    return launches


def job_logs():
    """{job_id: path} for every campaign job log."""
    out = {}
    for p in sorted(glob.glob(os.path.join(LOGS, "*.out"))):
        m = re.search(r"\.(\d{7})\.out$", p)
        if m:
            out[m.group(1)] = p
    return out


# ------------------------------------------------------------ Slurm extract
def _ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S") if s and s not in ("None", "Unknown") else None


def slurm_jobs():
    with open(os.path.join(PROV, "slurm_jobs.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("submit", "start", "end"):
            r[k + "_dt"] = _ts(r[k])
    return {r["job_id"]: r for r in rows}


def slurm_steps():
    with open(os.path.join(PROV, "slurm_steps.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["start_dt"] = _ts(r["start"])
        r["end_dt"] = _ts(r["end"])
    return rows


def submit_env(submit_line):
    """Variables exported by 'sbatch --export=ALL,K=V,...' (values may hold spaces)."""
    m = re.search(r"--export=(\S.*?)(?= -[A-Za-z]| \S+\.slurm)", submit_line)
    env = {}
    if not m:
        return env
    parts = m.group(1).split(",")
    key = None
    for p in parts:
        if "=" in p and re.match(r"^[A-Z_][A-Z0-9_]*=", p):
            key, val = p.split("=", 1)
            env[key] = val.strip('"')
        elif key:
            env[key] += "," + p
    return env


# ------------------------------------------------------------ versions
def version_timeline():
    """Rows of provenance/version_timeline.csv with 'uploaded' as a CDT datetime."""
    with open(os.path.join(PROV, "version_timeline.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["uploaded"] = datetime.strptime(r["uploaded_cdt"], "%Y-%m-%d %H:%M:%S")
    return rows


def version_live(timeline, filename, when):
    """The version of 'filename' on Stampede3 at time 'when' (CDT): the last
    upload that started before it. Uploads completed 3 to 6 s after starting."""
    cur = None
    for r in timeline:
        if r["file"] == filename and r["uploaded"] <= when:
            if cur is None or r["uploaded"] >= cur["uploaded"]:
                cur = r
    return cur


def seconds_from_nearest_upload(timeline, filename, when):
    ds = [abs((r["uploaded"] - when).total_seconds()) for r in timeline if r["file"] == filename]
    return min(ds) if ds else None


# ------------------------------------------------------------ step index
def interpolate(points, k):
    """Linear interpolation of [(step_index, value)] at step k; None outside
    the logged range (no extrapolation)."""
    pts = sorted(points)
    for (k0, v0), (k1, v1) in zip(pts, pts[1:]):
        if k0 <= k <= k1:
            return v0 if k1 == k0 else v0 + (v1 - v0) * (k - k0) / (k1 - k0)
    if pts and pts[0][0] == k:
        return pts[0][1]
    return None


def least_squares_slope(points):
    n = len(points)
    if n < 2:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    return sum((p[0] - mx) * (p[1] - my) for p in points) / sxx if sxx else None
