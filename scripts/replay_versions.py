# -*- coding: utf-8 -*-
"""Replay provenance/edit_history.json and check every uploaded version.

Each file's history starts with a full 'write'. A 'replace' applies exact
string replacements; count 1 requires the old text to occur exactly once. At
each 'upload', the current bytes are the version that reached Stampede3.

Checks, all of which must pass:
  1. each uploaded version equals, byte for byte, the copy stored under
     provenance/versions/ and the hash in provenance/version_timeline.csv;
  2. the first upload of each file equals the recovered 6 Sep bundle member,
     and the last upload equals the file read on Stampede3 on 2026-09-23.
Check 2 ties the two ends of every rebuilt chain to bytes read from TACC.
The Slurm allocation id is redacted to a same-length placeholder in the
edit history and in every stored version alike (provenance/README.md), so
both checks hold on the redacted bytes.

Writes processed/provenance_replay.txt. Exit status 1 on any mismatch.
"""
import hashlib
import json
import os
import sys

from common import PROCESSED, PROV, ROOT, version_timeline

REC_FINAL = os.path.join(PROV, "versions", "recovered", "stampede3_2026-09-23")
REC_TGZ = os.path.join(PROV, "versions", "recovered", "zmig_tgz_2026-09-06")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    hist = json.load(open(os.path.join(PROV, "edit_history.json"), encoding="utf-8"))
    timeline = version_timeline()
    by_file = {}
    for r in timeline:
        by_file.setdefault(r["file"], []).append(r)
    for rows in by_file.values():
        rows.sort(key=lambda r: r["uploaded"])

    state, uploads, out, bad = {}, {}, [], 0
    for e in hist["events"]:
        if e["action"] == "write":
            state[e["file"]] = e["content"]
        elif e["action"] == "replace":
            s = state[e["file"]]
            for rep in e["replacements"]:
                n = s.count(rep["old"])
                if n == 0 or (rep["count"] == 1 and n != 1):
                    raise SystemExit("event %d: old text occurs %d times in %s" % (e["seq"], n, e["file"]))
                s = s.replace(rep["old"], rep["new"]) if rep["count"] == "all" else s.replace(rep["old"], rep["new"], rep["count"])
            state[e["file"]] = s
        elif e["action"] == "upload":
            for f in e["files"]:
                if f in state:
                    uploads.setdefault(f, []).append((e["time_utc"], state[f].encode("utf-8")))

    out.append("Replay of provenance/edit_history.json (%d events)" % len(hist["events"]))
    for f, ups in sorted(uploads.items()):
        rows = by_file.get(f, [])
        if len(rows) != len(ups):
            out.append("MISMATCH %s: %d uploads replayed, %d versions in the timeline" % (f, len(ups), len(rows)))
            bad += 1
            continue
        for (t, b), r in zip(ups, rows):
            h = sha(b)
            stored = open(os.path.join(ROOT, r["path"]), "rb").read()
            ok = h == r["sha256"] == sha(stored) and t == r["uploaded_utc"]
            bad += not ok
            out.append("%-4s %-15s %-9s uploaded %s  sha256 %s  %s" % (
                "ok" if ok else "BAD", f, r["label"], t, h[:16], r["category"]))
        first, last = ups[0][1], ups[-1][1]
        tgz = os.path.join(REC_TGZ, f)
        fin = os.path.join(REC_FINAL, f)
        if os.path.exists(tgz):
            ok = first == open(tgz, "rb").read()
            bad += not ok
            out.append("%-4s %-15s first upload equals the recovered 6 Sep bundle member" % ("ok" if ok else "BAD", f))
        ok = last == open(fin, "rb").read()
        bad += not ok
        out.append("%-4s %-15s last upload equals the file read on Stampede3 on 2026-09-23" % ("ok" if ok else "BAD", f))
    for f in ("one_allreduce.py", "zmon.sh"):
        out.append("--   %-15s no edit history; the recovered original is the only version" % f)
    out.append("RESULT: %s" % ("all checks passed" if not bad else "%d checks FAILED" % bad))
    os.makedirs(PROCESSED, exist_ok=True)
    with open(os.path.join(PROCESSED, "provenance_replay.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out[-3:]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
